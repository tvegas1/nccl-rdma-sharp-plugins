/***************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 **************************************************************************/

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>
#include <x86intrin.h>

#include "nccl.h"
#include "net.h"

#define STATS_MAX_THREAD  128
#define ARRAY_SIZE(array) (sizeof(array) / sizeof(*(array)))

struct nccl_stat_entry {
  uint64_t min, max, total, count;
};

static inline uint64_t cycles() { return __rdtsc(); }

/* from timer.h */
static double coef() {
  volatile uint64_t total = 0;
  uint64_t c              = 0;
  struct timeval tv;
  double time;
  int iter = 100000;

  while (c == 0) {
    gettimeofday(&tv, NULL);
    c    = cycles();
    time = tv.tv_sec * 1e6 + tv.tv_usec;

    for (int i = 0; i < iter; i++) {
      total += __rdtsc();
    }

    gettimeofday(&tv, NULL);
    c     = cycles() - c;
    time  = tv.tv_sec * 1e6 + tv.tv_usec - time;
    iter *= 10;
  }
  return time / c;
}

struct nccl_statistics {
  struct {
    uint64_t isend;
    uint64_t irecv;
    uint64_t iflush;
    uint64_t iflush_miss;
    uint64_t irecv_miss;
    uint64_t isend_miss;
    uint64_t test;
  } call; /* Call count */

  struct {
    struct nccl_stat_entry isend;
  } sizes;

  struct {
    struct nccl_stat_entry isend;
    struct nccl_stat_entry irecv;
    struct nccl_stat_entry iflush;
    struct nccl_stat_entry test;
  } durations;

  struct nccl_stat_entry irecv_n; /* total irecv entries posted */

} __attribute__((aligned(64)));

struct nccl_stats_plugin {
  union {
    ncclNet_v6_t v6;
  } plugin;

  struct nccl_statistics stats[STATS_MAX_THREAD];
  int                    stats_count;
};

enum stats_req_type { REQ_IFLUSH, REQ_IRECV, REQ_ISEND };

struct stats_req {
  void                *request; /* original request */

  enum stats_req_type type;
  uint64_t            start; /* cycles */
};

static struct nccl_stats_plugin ctx             = {};
static __thread struct stats_req *req_free_list = NULL;

static void nccl_stat_entry_add(struct nccl_stat_entry *entry, uint64_t val) {
  entry->total += val;
  entry->count++;

  if (entry->min == 0 || entry->min > val) {
    entry->min = val;
  }

  if (entry->max == 0 || entry->max < val) {
    entry->max = val;
  }
}

static void nccl_stats_entry_dump_coef(struct nccl_stat_entry *entry,
                                       double coef, const char *name) {
  printf("%s %" PRIu64 " %.2lf/%.2lf/%.2lf", name, entry->count,
         entry->min * coef, entry->max * coef,
         entry->count ? entry->total * coef / entry->count : 0);
}

static void nccl_stats_entry_dump(struct nccl_stat_entry *entry,
                                  const char *name) {
  printf("%s %" PRIu64 " %" PRIu64 "/%" PRIu64 "/%" PRIu64, name, entry->count,
         entry->min, entry->max,
         entry->count ? entry->total / entry->count : 0);
}

static struct nccl_statistics *stats(void) {
  static __thread struct nccl_statistics *stats = NULL;

  if (!stats) {
    int index = __sync_fetch_and_add(&ctx.stats_count, 1);
    if (index < ARRAY_SIZE(ctx.stats)) {
      stats = &ctx.stats[index];
    }
  }

  assert(stats != NULL);
  return stats;
}

static struct stats_req *get(void) {
  struct stats_req *req;

  req = req_free_list;
  if (req != NULL) {
    req_free_list = *(struct stats_req**)req;
  } else {
    req = malloc(sizeof(*req));
    assert(req != NULL);
  }

  return req;
}

static void put(struct stats_req *req) {
  *(struct stats_req**)req = req_free_list;
  req_free_list            = req;
}

static struct stats_req *req_create(void *request, enum stats_req_type type,
                                    uint64_t cycles) {
  struct stats_req *req = get();

  req->request = request;
  req->type    = type;
  req->start   = cycles;
  return req;
}

static void stats_isend(size_t size, void **request, uint64_t cycles) {
  /* a request was created */
  if (*request) {
    *request = req_create(*request, REQ_ISEND, cycles);
    stats()->call.isend++;
    nccl_stat_entry_add(&stats()->sizes.isend, size);
  } else {
    stats()->call.isend_miss++;
  }
}

static void stats_irecv(int n, int *sizes, void **request, uint64_t cycles) {
  /* a receive request was created */
  if (*request) {
    *request = req_create(*request, REQ_IRECV, cycles);
    nccl_stat_entry_add(&stats()->irecv_n, n);
    stats()->call.irecv++;
  } else {
    stats()->call.irecv_miss++;
  }
}

static void stats_iflush(int n, int *sizes, void **request, uint64_t cycles) {
  if (*request) {
    *request = req_create(*request, REQ_IFLUSH, cycles);
    stats()->call.iflush++;
  } else {
    stats()->call.iflush_miss++;
  }
}

static void stats_test(int *done, void *request, uint64_t start) {
  struct stats_req *req = request;
  struct nccl_stat_entry *entry;

  if (*done) {
    if (req->type == REQ_IRECV) {
      entry = &stats()->durations.irecv;
    } else if (req->type == REQ_ISEND) {
      entry = &stats()->durations.isend;
    } else {
      assert(req->type == REQ_IFLUSH);
      entry = &stats()->durations.iflush;
    }

    nccl_stat_entry_add(entry, cycles() - req->start);
    nccl_stat_entry_add(&stats()->durations.test, cycles() - start);
    put(req);
  }

  stats()->call.test++;
}

static ncclResult_t nccl_stats_isend_v6(void *send_comm, void *data, int size,
                                        int tag, void *mhandle,
                                        void **request) {
  uint64_t start = cycles();
  ncclResult_t result =
      ctx.plugin.v6.isend(send_comm, data, size, tag, mhandle, request);
  stats_isend(size, request, start);
  return result;
}

static ncclResult_t nccl_stats_irecv_v6(void *recv_comm, int n, void **data,
                                        int *sizes, int *tags, void **mhandle,
                                        void **request) {
  uint64_t start = cycles();
  ncclResult_t result =
      ctx.plugin.v6.irecv(recv_comm, n, data, sizes, tags, mhandle, request);
  stats_irecv(n, sizes, request, start);
  return result;
}

static ncclResult_t nccl_stats_iflush_v6(void *recv_comm, int n, void **data,
                                         int *sizes, void **mhandle,
                                         void **request) {
  uint64_t start = cycles();
  ncclResult_t result =
      ctx.plugin.v6.iflush(recv_comm, n, data, sizes, mhandle, request);
  stats_iflush(n, sizes, request, start);
  return result;
}

static ncclResult_t nccl_stats_test_v6(void *request, int *done, int *size) {
  struct stats_req *req = request;
  uint64_t start        = cycles();

  ncclResult_t result = ctx.plugin.v6.test(req->request, done, size);
  stats_test(done, request, start);
  return result;
}

static void __attribute__((destructor)) nccl_stats_dump(void) {
  double c  = coef();
  pid_t pid = getpid();

  for (int i = 0; i < ctx.stats_count; i++) {
    printf("%d/#%d CALL"
           " isend %" PRIu64 "/%" PRIu64 " irecv %" PRIu64 "/%" PRIu64
           " iflush %" PRIu64 "/%" PRIu64 " test %" PRIu64,
           pid, i, ctx.stats[i].call.isend, ctx.stats[i].call.isend_miss,
           ctx.stats[i].call.irecv, ctx.stats[i].call.irecv_miss,
           ctx.stats[i].call.iflush, ctx.stats[i].call.iflush_miss,
           ctx.stats[i].call.test);
    printf(" ");
    nccl_stats_entry_dump(&ctx.stats[i].irecv_n, "irecv_n");
    printf(" ");
    nccl_stats_entry_dump(&ctx.stats[i].sizes.isend, "isend");
    printf(" DUR ");
    nccl_stats_entry_dump_coef(&ctx.stats[i].durations.irecv, c, "irecv");
    printf(" ");
    nccl_stats_entry_dump_coef(&ctx.stats[i].durations.isend, c, "isend");
    printf(" ");
    nccl_stats_entry_dump_coef(&ctx.stats[i].durations.iflush, c, "iflush");
    printf("\n");
  }
}

ncclNet_v6_t ncclStatsInit_v6(const ncclNet_v6_t *plugin) {
  static char name[32] = "STATS-unknown";
  ncclNet_v6_t s       = *plugin; /* Reuse all members */

  snprintf(name, sizeof(name), "STATS-%s", plugin->name);

  s.name   = name;
  s.isend  = nccl_stats_isend_v6;
  s.irecv  = nccl_stats_irecv_v6;
  s.iflush = nccl_stats_iflush_v6;
  s.test   = nccl_stats_test_v6;

  ctx.plugin.v6 = *plugin;

  return s;
}
