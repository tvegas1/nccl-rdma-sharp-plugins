/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ucx_uct_lib.h"

typedef enum {
  NCCL_UCT_ISEND = 1,
  NCCL_UCT_IRECV,
  NCCL_UCT_IFLUSH
} nccl_uct_req_type_t;

typedef enum {
  NCCL_UCT_NONE = 0,
  NCCL_UCT_ATP,
  NCCL_UCT_ATP_COMPLETE
} nccl_uct_ack_mode_;

/* On the wire ack message */
typedef struct nccl_uct_atp {
  int               sizes[NCCL_UCX_UCT_MAX_RECVS];
  unsigned          rtr_id;     /* Id to match against idx */
  short             count;      /* Original number of receives */
  short             start;      /* How many left to start */
  short             complete;   /* How many left to complete */
  short             send_atp;
  unsigned          idx;        /* Will match rtr_id when received */
} nccl_uct_atp_t;

/* On the wire memory chunk to send to */
typedef struct nccl_uct_chunk {
    void            *data;
    int             size;
    int             tag;
    uct_rkey_t      rkey;
} nccl_uct_chunk_t;

/* On the wire request message */
typedef struct {
    nccl_uct_chunk_t chunk[NCCL_UCX_UCT_MAX_RECVS];
    unsigned         count;         /* Total number of chunks */
    unsigned         avail;         /* Total number of chunk left */
    unsigned         send_atp;      /* 0 no send, 1 send, 2 complete-send */
    unsigned         id;            /* Id of the receive */
} nccl_uct_rtr_t;

/* Track the sending of a request */
typedef struct {
    uct_completion_t        completion; /* Completion for put and others */
    struct nccl_uct_wr_comm *comm;      /* Owning communicator */
    nccl_uct_req_type_t     type;       /* Type of the request */
    unsigned                id;         /* RTR id of the request (not for flush) */
    unsigned                size;       /* Size of the send request (isend) */
    unsigned                idx;
} nccl_uct_req_t;

#define NCCL_UCT_RING_SIZE (128 * 4)    /* TODO: Add build static checks */
#define NCCL_UCT_RING_MASK (NCCL_UCT_RING_SIZE - 1)

#if 0
#define TSHOOT(...) WARN(__VA_ARGS__)
#else
#define TSHOOT(...)
#endif


typedef struct nccl_uct_wr_comm {
  nccl_uct_comm_t      base;

  nccl_uct_atp_t       atp[NCCL_UCT_RING_SIZE];
  nccl_uct_rtr_t       rtr[NCCL_UCT_RING_SIZE];

  nccl_uct_memh_t      *atp_memh;
  nccl_uct_memh_t      *rtr_memh;

  /* NCCL request for isend, irecv or iflush */
  nccl_uct_req_t       req[NCCL_UCT_RING_SIZE];

  unsigned             rtr_id; /* Next irecv request position to allocate */
  unsigned             req_id; /* Next NCCL request to allocate */

  unsigned             total;  /* Request in progress */
} nccl_uct_wr_comm_t;

static inline nccl_uct_wr_comm_t *
nccl_uct_wr_comm_get(nccl_uct_comm_t *base_comm) {
  return ucs_container_of(base_comm, nccl_uct_wr_comm_t, base);
}

/* On receiver side, after ->irecv(), expect corresponding ATP */
static ucs_status_t nccl_uct_atp_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
  nccl_uct_wr_comm_t *comm = nccl_uct_wr_comm_get(*(nccl_uct_comm_t **)data);
  nccl_uct_atp_t *src_atp  = (nccl_uct_atp_t*)((uint8_t*)data + 8);

  assert(length == (sizeof(*src_atp) + 8));
  memcpy(&comm->atp[src_atp->rtr_id & NCCL_UCT_RING_MASK], src_atp, sizeof(*src_atp));
  return UCS_OK;
}

static ncclResult_t nccl_uct_wr_iface_set(nccl_uct_iface_t *uct_iface) {
  NCCLCHECK(nccl_uct_iface_set_handler(uct_iface, NCCL_UCT_AM_ATP,
                                       nccl_uct_atp_callback));
  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_comm_alloc(nccl_uct_comm_t **comm_p) {
  nccl_uct_wr_comm_t *comm = calloc(1, sizeof(nccl_uct_wr_comm_t));
  if (comm != NULL) {
    *comm_p = &comm->base;
    return ncclSuccess;
  }

  return ncclSystemError;
}

static ncclResult_t nccl_uct_wr_comm_init(nccl_uct_comm_t *base_comm,
                                          nccl_uct_context_t *context,
                                          nccl_uct_worker_t *worker, int dev,
                                          const nccl_uct_comm_t *remote_comm) {
  nccl_uct_wr_comm_t *comm = nccl_uct_wr_comm_get(base_comm);

  NCCLCHECK(nccl_uct_comm_init(&comm->base, context, worker, dev, remote_comm));
  NCCLCHECK(nccl_uct_reg_mr(comm, comm->rtr, sizeof(comm->rtr), 0, (void **)&comm->rtr_memh));
  NCCLCHECK(nccl_uct_reg_mr(comm, comm->atp, sizeof(comm->atp), 0, (void **)&comm->atp_memh));

  comm->base.remote.addr.rtr_ptr  = comm->rtr;
  comm->base.remote.addr.rtr_rkey = comm->rtr_memh->bundle.rkey;
  comm->base.remote.addr.atp_ptr  = comm->atp;
  comm->base.remote.addr.atp_rkey = comm->atp_memh->bundle.rkey;

  comm->rtr_id = 1;
  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_init(ncclDebugLogger_t logFunction) {
  context.ops.comm_alloc = nccl_uct_wr_comm_alloc;
  context.ops.comm_init  = nccl_uct_wr_comm_init;
  context.ops.iface_set  = nccl_uct_wr_iface_set;
  context.am_short_size  = sizeof(nccl_uct_atp_t);
  context.rkey_size      = sizeof(uct_rkey_t);

  return nccl_p2p_ib_init(&context.dev_count, ncclIbDevs, context.if_name,
                          &context.if_addr, NULL, logFunction);
}

static ncclResult_t nccl_uct_put(nccl_uct_wr_comm_t *comm,
                                 void *data, size_t size,
                                 nccl_uct_memh_t *uct_memh,
                                 void *rva, uct_rkey_t rkey,
                                 uct_completion_t *completion)
{
    ucs_status_t status;
    uct_iov_t iov;

    iov.buffer = data;
    iov.length = size;
    iov.memh   = uct_memh->memh;
    iov.stride = 0;
    iov.count  = 1;

    status = uct_ep_put_zcopy(comm->base.uct_ep->ep, &iov, 1,
                              (uint64_t)rva, rkey, completion);
    if (status == UCS_OK) {
        completion->count--;
    } else if (status != UCS_INPROGRESS) {
        return UCS_ERR_NO_RESOURCE;
    }

    return status;
}

static ncclResult_t nccl_uct_wr_irecv(void *recv_comm, int n, void **data,
                                      int *sizes, int *tags, void **mhandles,
                                      void **request) {
  nccl_uct_wr_comm_t *comm   = nccl_uct_wr_comm_get(recv_comm);
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandles;
  int i, idx, end = NCCL_UCX_UCT_MAX_RECVS;
  nccl_uct_rtr_t *rtr;
  volatile nccl_uct_atp_t *atp;
  nccl_uct_req_t *req;
  ucs_status_t status;

  assert(n <= end);

  rtr           = &comm->rtr[comm->rtr_id & NCCL_UCT_RING_MASK];
  rtr->count    = n;
  rtr->avail    = n;
  rtr->send_atp = NCCL_UCT_ATP;
  rtr->id       = comm->rtr_id;

  atp         = &comm->atp[comm->rtr_id & NCCL_UCT_RING_MASK];
  atp->rtr_id = rtr->id;
  atp->idx    = rtr->id;
  atp->count  = n;

  if (rtr->send_atp) {
      atp->idx -= 0x01010101; /* Make it different */
      memset((void *)atp->sizes, 0, sizeof(atp->sizes));
  } else {
      memcpy((void *)(atp->sizes + end - n), sizes, n * sizeof(*sizes));
  }

  idx = end - n;
  for (i = 0; i < n; i++, idx++) {
      rtr->chunk[idx].data = data[i];
      rtr->chunk[idx].size = sizes[i];
      rtr->chunk[idx].tag  = tags[i];
      rtr->chunk[idx].rkey = uct_memh[i]->bundle.rkey;
  }

  req                    = &comm->req[comm->req_id & NCCL_UCT_RING_MASK];
  req->completion.func   = nccl_uct_empty_callback;
  req->completion.count  = 1;
  req->completion.status = UCS_OK;
  req->type              = NCCL_UCT_IRECV;
  req->id                = comm->rtr_id;
  req->comm              = comm;

  status = nccl_uct_put(comm, rtr, sizeof(*rtr), comm->rtr_memh,
                        (nccl_uct_rtr_t *)comm->base.remote.addr.rtr_ptr + (comm->rtr_id & NCCL_UCT_RING_MASK),
                        comm->base.remote.addr.rtr_rkey,
                        &req->completion);
  if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
      *request = req;
      TSHOOT("irecv send n=%d req_id=%u rtr_id=%u", n, comm->req_id, comm->rtr_id);
      comm->req_id++;
      comm->total++;
      comm->rtr_id++;
  } else {
      *request = NULL;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_send(nccl_uct_wr_comm_t *comm, unsigned id,
                                  unsigned i, void *data, int size,
                                  nccl_uct_memh_t *uct_memh, void **request)
{
    int end                      = NCCL_UCX_UCT_MAX_RECVS;
    volatile nccl_uct_rtr_t *rtr = &comm->rtr[id & NCCL_UCT_RING_MASK];
    nccl_uct_atp_t *atp          = &comm->atp[id & NCCL_UCT_RING_MASK];
    volatile nccl_uct_chunk_t *chunk      = &rtr->chunk[end - rtr->count + i];
    nccl_uct_req_t *req;
    ucs_status_t status;

    req                    = &comm->req[comm->req_id & NCCL_UCT_RING_MASK];
    req->completion.func   = nccl_uct_empty_callback;
    req->completion.count  = 3;
    req->completion.status = UCS_OK;
    req->type              = NCCL_UCT_ISEND;
    req->id                = id;
    req->size              = size;
    req->comm              = comm;
    req->idx               = i;

    assert(size <= chunk->size);

    status = nccl_uct_put(comm, data, size, uct_memh,
                          chunk->data, chunk->rkey, &req->completion);
    if ((status != UCS_OK) && (status != UCS_INPROGRESS)) {
        *request = NULL;
        return ncclSuccess;
    }

    if (rtr->avail == rtr->count) {
        TSHOOT("%p isend new req_id=%u count=%u send_atp=%u atp=%p",
             comm, rtr->id, rtr->count, rtr->send_atp, atp);
        atp->rtr_id   = rtr->id;
        atp->count    = rtr->count;
        atp->start    = rtr->count;
        atp->complete = rtr->count;
        atp->send_atp = rtr->send_atp;
        atp->idx      = rtr->id;
    } 

    TSHOOT("%p isend started req=%p req->id=%u idx=%u/%u size=%d", req, comm, id, i, rtr->count, size);

    assert(req->id == rtr->id);
    assert(atp->send_atp);
    assert(atp->rtr_id == rtr->id);
    assert(rtr->count == atp->count);
    assert(rtr->avail > 0);

    atp->sizes[end - rtr->count + i] = size;
    atp->start--;
    rtr->avail--;
    chunk->tag = INT_MAX;

    comm->req_id++;
    comm->total++;
    *request = req;
    return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_isend(void *send_comm, void *data, int size,
                                      int tag, void *mhandle, void **request) {
  nccl_uct_wr_comm_t *comm = nccl_uct_wr_comm_get(send_comm);
  volatile nccl_uct_rtr_t *rtr;
  int id, i;
  unsigned end = NCCL_UCX_UCT_MAX_RECVS;

  for (id = comm->rtr_id;; id++) {
      rtr = &comm->rtr[id & NCCL_UCT_RING_MASK];
      if (rtr->id != id) {
          break;
      }

      __sync_synchronize(); /* TODO remove some synchronize? */

      if (rtr->avail == 0) {
          if (id == comm->rtr_id) {
              comm->rtr_id++;
          }
          continue;
      }
      TSHOOT("isend got id=%u n=%u", id, rtr->count);

      for (i = 0; i < rtr->count; i++) {
          if (rtr->chunk[end - rtr->count + i].tag == tag) {
              return nccl_uct_send(comm, id, i, data, size, mhandle, request);
          }
      }
  }

  *request = NULL;
  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_iflush(void *recv_comm, int n, void **data,
                                       int *sizes, void **mhandle,
                                       void **request) {
  nccl_uct_wr_comm_t *comm   = nccl_uct_wr_comm_get(recv_comm);
  int last                   = nccl_uct_flush_index(&comm->base, sizes, n);
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandle;
  nccl_uct_req_t *req;

  if (last == -1) {
    *request = NULL;
    return ncclSuccess;
  }

  /* Consume request anyways */
  req                    = &comm->req[comm->req_id++ & NCCL_UCT_RING_MASK];
  req->completion.func   = nccl_uct_empty_callback;
  req->completion.count  = 1;
  req->completion.status = UCS_OK;
  req->type              = NCCL_UCT_IFLUSH;
  req->comm              = comm;
  *request               = req;
  comm->total++;

  return nccl_uct_flush(&comm->base, data[last], sizes[last], uct_memh[last],
                        &req->completion, request);
}

static void nccl_uct_wr_send_atp(nccl_uct_wr_comm_t *comm, nccl_uct_req_t *req,
                                 nccl_uct_atp_t *atp)
{
    ucs_status_t status;

    /*
     * No send: just finish all sends
     * ATP    : one isend will track 
     * ATP_PUT: reuse completion
     */

    assert(req->id == atp->rtr_id);

    if (atp->send_atp == NCCL_UCT_ATP && (atp->start == 0)) {
        status = uct_ep_fence(comm->base.uct_ep->ep, 0);
        assert(status == UCS_OK);

        status = uct_ep_am_short(comm->base.uct_ep->ep, NCCL_UCT_AM_ATP,
                                 (uint64_t)comm->base.remote.comm, atp, sizeof(*atp));
        if (status == UCS_OK) {
            atp->send_atp = 0;
        }
        return;
    }

    if ((atp->send_atp == NCCL_UCT_ATP_COMPLETE) && (atp->complete == 0)) {
        req->completion.count = 1;
        req->completion.status = UCS_OK;

        /* TODO: Do we really need fence */
        status = uct_ep_fence(comm->base.uct_ep->ep, 0);
        assert(status == UCS_OK);

        TSHOOT("%p atp_put req->id=%u", comm, req->id);
        status = nccl_uct_put(comm, atp, sizeof(*atp), comm->atp_memh,
                              (nccl_uct_atp_t *)comm->base.remote.addr.atp_ptr + (req->id & NCCL_UCT_RING_MASK),
                              comm->base.remote.addr.atp_rkey,
                              &req->completion);
        if (status == UCS_ERR_NO_RESOURCE) { 
            req->completion.count = 0;
        } else {
            atp->send_atp = 0;
        }
    }
}

// TODO Fix barriers here too
static ncclResult_t nccl_uct_wr_test(void *request, int *done, int *sizes) {
  nccl_uct_req_t *req      = request;
  nccl_uct_wr_comm_t *comm = req->comm;
  int end                  = NCCL_UCT_RING_MASK;
  volatile nccl_uct_atp_t *atp;

  while (uct_worker_progress(comm->base.uct_worker->worker)) {}

  if (req->type == NCCL_UCT_IRECV) {
      if (req->completion.count > 0) {
          *done = 0;
          return ncclSuccess;
      }

      atp = &comm->atp[req->id & NCCL_UCT_RING_MASK];
      assert(atp->rtr_id == req->id);
      assert(comm->rtr[req->id & NCCL_UCT_RING_MASK].id == req->id);

      *done = (atp->idx == atp->rtr_id);
      if (*done) {
          __sync_synchronize();
          TSHOOT("%p irecv rtr_id=%u completed", comm, req->id);
          memcpy(sizes, (void *)&atp->sizes[end - atp->count],
                 atp->count * sizeof(*sizes));
      }
  } else if (req->type == NCCL_UCT_ISEND) {
      atp = &comm->atp[req->id & NCCL_UCT_RING_MASK];

      assert(atp->rtr_id == req->id);
      assert(atp->idx == req->id);

      if (req->completion.count == 2) {
          atp->complete--; /* actual isend is done */
          req->completion.count = 0;
          TSHOOT("%p isend req->id=%u put completed", comm, req->id);
      }

      nccl_uct_wr_send_atp(comm, req, (nccl_uct_atp_t *)atp);

      if ((req->completion.count == 0) && (!atp->send_atp || atp->complete)) {
          TSHOOT("%p isend req->id=%u i=%u done", comm, req->id, req->idx);
          *done = 1;
          if (sizes) {
              *sizes = req->size;
          }
      } else {
          *done = 0;
      }
  } else {
      assert(req->type == NCCL_UCT_IFLUSH);
      *done = (req->completion.count == 0);
  }

  if (*done) {
      comm->total--;
  }
  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_close(void *close_comm) {
  nccl_uct_wr_comm_t *comm = nccl_uct_wr_comm_get(close_comm);

  nccl_uct_dereg_mr(comm, comm->rtr_memh);
  nccl_uct_dereg_mr(comm, comm->atp_memh);

  nccl_uct_comm_deinit(close_comm);

  assert(comm->total == 0);
  /* TODO add asserts */
  free(comm);
  return ncclSuccess;
}

ncclNet_v8_t ucxUctPlugin_v8 = NCCL_UCT_PLUGIN_V8("UCX-UCT", nccl_uct_wr);
ncclNet_v7_t ucxUctPlugin_v7 = NCCL_UCT_PLUGIN_V7("UCX-UCT", nccl_uct_wr);
ncclNet_v6_t ucxUctPlugin_v6 = NCCL_UCT_PLUGIN_V6("UCX-UCT", nccl_uct_wr);
ncclNet_v5_t ucxUctPlugin_v5 = NCCL_UCT_PLUGIN_V5("UCX-UCT", nccl_uct_wr);
