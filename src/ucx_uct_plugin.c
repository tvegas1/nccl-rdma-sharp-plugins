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

/* On the wire ack message */
typedef struct nccl_uct_atp {
  int               sizes[NCCL_UCX_UCT_MAX_RECVS];
  unsigned          rtr_id;
  int               count;      /* Original number of receives */
  short             start;      /* How many left to start */
  short             complete;   /* How many left to complete */
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
    unsigned         count;
    unsigned         avail;
    unsigned         send_atp;
    unsigned         id;
} nccl_uct_rtr_t;

/* Track the sending of a request */
typedef struct {
    uct_complete_t      complete;
    nccl_uct_req_type_t type;
    unsigned            id;
    unsigned            size;
} nccl_uct_req_t;

#define NCCL_UCT_RING_SIZE (MAX_REQUESTS * 4)

typedef struct nccl_uct_wr_comm {
  nccl_uct_comm_t      base;

  nccl_uct_atp_t       atp[NCCL_UCT_RING_SIZE];
  nccl_uct_rtr_t       rtr[NCCL_UCT_RING_SIZE];

  nccl_uct_memh_t      atp_memh;
  nccl_uct_memh_t      rtr_memh;

  /* NCCL request for isend, irecv or iflush */
  nccl_uct_req_t       req[NCCL_UCT_RING_SIZE];

  unsigned             rtr_id;  /* Next irecv request to allocate */
  unsigned             req_id; /* Next request to allocate */
} nccl_uct_wr_comm_t;

static inline nccl_uct_wr_comm_t *
nccl_uct_wr_comm_get(nccl_uct_comm_t *base_comm) {
  return ucs_container_of(base_comm, nccl_uct_wr_comm_t, base);
}

/* On receiver side, after ->irecv(), expect corresponding ATP */
static ucs_status_t nccl_uct_atp_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
    /* TODO implement */
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
  NCCLCHECK(nccl_uct_reg_mr(comm, comm->rtr, sizeof(comm->rtr), 0, &comm->rtr_memh));
  NCCLCHECK(nccl_uct_reg_mr(comm, comm->atp, sizeof(comm->atp), 0, &comm->atp_memh));

  comm->remote.addr.rtr_ptr  = comm->rtr;
  comm->remote.addr.rtr_rkey = comm->rtr_memh->bundle.rkey;
  comm->remote.addr.atp_ptr  = comm->atp;
  comm->remote.addr.atp_rkey = comm->atp_memh->bundle.rkey;

  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_init(ncclDebugLogger_t logFunction) {
  context.ops.comm_alloc = nccl_uct_wr_comm_alloc;
  context.ops.comm_init  = nccl_uct_wr_comm_init;
  context.ops.iface_set  = nccl_uct_wr_iface_set;
  context.am_short_size  = nccl_uct_rdesc_size(NCCL_UCX_UCT_MAX_RECVS);
  context.rkey_size      = sizeof(nccl_uct_atp_t);

  return nccl_p2p_ib_init(&context.dev_count, ncclIbDevs, context.if_name,
                          &context.if_addr, NULL, logFunction);
}

/* Outcome is either send_atp equal to 1 or 0 */
static void nccl_uct_send_atp(nccl_uct_wr_comm_t *comm,
                              nccl_uct_rdesc_t *rdesc) {
  ucs_status_t status;
  nccl_uct_atp_t atp;
  int i;

  assert(rdesc->send_atp == 1);

  status = uct_ep_fence(comm->base.uct_ep->ep, 0);
  if (status != UCS_OK) {
    return;
  }

  atp.id    = rdesc->desc.id;
  atp.rdesc = rdesc->desc.peer_rdesc;
  atp.count = rdesc->desc.count;

  /* Sizes from isend() are lower or equal to their irecv() side */
  for (i = 0; i < rdesc->desc.count; i++) {
    atp.sizes[i] = rdesc->reqs[i].size;
  }

  status = uct_ep_am_short(comm->base.uct_ep->ep, NCCL_UCT_AM_ATP,
                           (uint64_t)comm->base.remote.comm, &atp, sizeof(atp));
  if (status == UCS_OK) {
    rdesc->send_atp = 0;
  }
}

static ncclResult_t nccl_uct_put(nccl_uct_wr_comm_t *comm,
                                 void *data, size_t size,
                                 nccl_uct_memh_t *uct_memh,
                                 void *rva, uct_rkey_h rkey,
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
                              rva, rkey, completion);
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
  int i, end = NCCL_UCX_UCT_MAX_RECVS;
  nccl_uct_rtr_t *rtr;
  volatile nccl_uct_atp_t *atp;
  nccl_uct_req_t *req;
  ucs_status_t status;

  assert(n <= end);

  rtr           = &comm->rtr[comm->rtr_id & NCCL_UCT_RING_SIZE];
  rtr->id       = comm->rtr_id + 1;
  rtr->count    = n;
  rtr->avail    = n;
  rtr->send_atp = 1;

  atp         = &comm->atp[comm->rtr_id & NCCL_UCT_RING_SIZE];
  atp->rtr_id = rtr->id;
  atp->idx    = rtr->id - 1;

  idx = end - n;
  for (i = 0; i < n; i++, idx++) {
      rtr->chunk[idx + i].data = data[i];
      rtr->chunk[idx + i].size = sizes[i];
      rtr->chunk[idx + i].tag  = tags[i];
      rtr->chunk[idx + i].rkey = uct_memh[i].bundle.rkey;
  }

  req                    = &comm->req[comm->req_id & NCCL_UCT_RING_SIZE];
  req->completion.func   = nccl_uct_empty_callback;
  req->completion.count  = 1;
  req->completion.status = UCS_OK;
  req->type              = NCCL_UCT_REQ_IRECV;
  req->id                = comm->rtr_id;

  status = nccl_uct_put(comm, rtr, sizeof(*rtr), comm->rtr_memh,
                        (nccl_uct_rtr_t *)comm->remote.addr.rtr_ptr + comm->rtr_id,
                        comm->remote.addr.rtr_rkey,
                        &req->completion);
  if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
      *request = req;
      comm->req_id++;
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
    volatile nccl_uct_rtr_t *rtr = &comm->rtr[id & NCCL_UCT_RING_SIZE];
    nccl_uct_atp_t *atp          = &comm->atp[id & NCCL_UCT_RING_SIZE];
    nccl_uct_chunk_t *chunk      = rtr->chunk[end - rtr->count + i];
    ucs_status_t status;

    req                    = &comm->req[comm->req_id & NCCL_UCT_RING_SIZE];
    req->completion.func   = nccl_uct_empty_callback;
    req->completion.count  = 2;
    req->completion.status = UCS_OK;
    req->type              = NCCL_UCT_REQ_ISEND;
    req->size              = size;
    req->id                = id;

    assert(size <= chunk->size);

    status = nccl_uct_put(comm, data, size, uct_memh,
                          chunk->data, chunk->rkey, &req->completion);
    if ((status != UCS_OK) && (status != UCS_INPROGRESS)) {
        *request = NULL;
        return ncclSuccess;
    }

    if (rtr->avail == rtr->count) {
        atp->rtr_id   = rtr->id;
        atp->count    = rtr->count;
        atp->start    = rtr->count;
        atp->complete = rtr->count;
        atp->send_atp = rtr->send_atp;
        atp->idx      = rtr->id;
    } 

    assert(atp->rtr_id == rtr->id);

    atp->sizes[end - rtr->count + i] = size;
    atp->start--;
    rtr->avail--;
    chunk->tag = INT_MAX;

    comm->req_id++;
    *request = req;
    return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_isend(void *send_comm, void *data, int size,
                                      int tag, void *mhandle, void **request) {
  nccl_uct_wr_comm_t *comm = nccl_uct_wr_comm_get(send_comm);
  volatile nccl_uct_rtr_t *rtr;
  int id, i, end = NCCL_UCX_UCT_MAX_RECVS;
  unsigned end;

  for (id = comm->rtr_id;; id++) {
      rtr = &comm->rtr[id & NCCL_UCT_RING_SIZE];
      if (rtr->id != (id + 1)) {
          break;
      }

      __sync_synchronize(); /* TODO remove some synchronize? */

      if (rtr->avail == 0) {
          if (id == comm->rtr_id) {
              comm->rtr_id++;
          }
          continue;
      }

      for (i = 0; i < rtr->count; i++) {
          if ((rtr->chunk[end - rtr->count + i].tag) == tag) {
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
  int last                   = nccl_uct_flush_index(base_comm, sizes, n);
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandle;
  nccl_uct_req_t *req;

  if (last == -1) {
    *request = NULL;
    return ncclSuccess;
  }

  req                    = &comm->req[comm->req_id & NCCL_UCT_RING_SIZE];
  req->completion.func   = nccl_uct_empty_callback;
  req->completion.count  = 1;
  req->completion.status = UCS_OK;
  req->type              = NCCL_UCT_REQ_IFLUSH;
  *request               = req;

  return nccl_uct_flush(base_comm, data[last], sizes[last], uct_memh[last],
                        &req->completion, request);
}

// TODO Fix barriers here too
static ncclResult_t nccl_uct_wr_test(void *request, int *done, int *sizes) {
  nccl_uct_req_t *req      = request;
  nccl_uct_rdesc_t *rdesc  = req->rdesc;
  nccl_uct_wr_comm_t *comm = rdesc->comm;
  volatile nccl_uct_atp_t *atp;

  while (uct_worker_progress(comm->base.uct_worker->worker)) {}

  if (req->type == NCCL_UCT_REQ_IRECV) {
      atp = &comm->atp[req->id & NCCL_UCT_RING_SIZE];
      assert(atp->rtr_id == (req->id + 1));

      if (atp->idx == atp->rtr_id) {
          __sync_synchronize();
          *done = 1;
          if (sizes != NULL) {
              memcpy(sizes, &atp->sizes[end - atp->count], atp->count);
          }
      }
  } else if (req->type == NCCL_UCT_REQ_ISEND) {
      // put completed
      // atp/atp_put ocmpleted?
      // is it the last one?
  } else {
      assert(req->type == NCCL_UCT_REQ_IFLUSH);

      *done = (req->completion.count == 0);
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_close(void *close_comm) {
  nccl_uct_wr_comm_t *comm = nccl_uct_wr_comm_get(close_comm);

  nccl_uct_dereg_mr(comm, comm->rtr_memh);
  nccl_uct_dereg_mr(comm, comm->atp_memh);

  nccl_uct_comm_deinit(close_comm);

  /* TODO add asserts */
  free(comm);
  return ncclSuccess;
}

ncclNet_v8_t ucxUctPlugin_v8 = NCCL_UCT_PLUGIN_V8("UCX-UCT", nccl_uct_wr);
ncclNet_v7_t ucxUctPlugin_v7 = NCCL_UCT_PLUGIN_V7("UCX-UCT", nccl_uct_wr);
ncclNet_v6_t ucxUctPlugin_v6 = NCCL_UCT_PLUGIN_V6("UCX-UCT", nccl_uct_wr);
ncclNet_v5_t ucxUctPlugin_v5 = NCCL_UCT_PLUGIN_V5("UCX-UCT", nccl_uct_wr);
