/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdint.h>
#include <unistd.h>

#include "p2p_plugin.h"
#include "socket.h"

#include <uct/api/uct.h>

#define NCCL_UCX_UCT_MAX_RECVS       NCCL_NET_IB_MAX_RECVS
#define NCCL_UCT_LISTEN_HANDLE_MAGIC 0x43cf19ed91abdb85
#define NCCL_UCT_REG_ALIGN           4096

/*
TODO: TRY
 120   {"TX_CQ_MODERATION", "64",
 121    "Maximum number of send WQEs which can be posted without requesting a completion.",
 122    ucs_offsetof(uct_rc_iface_config_t, tx_cq_moderation), UCS_CONFIG_TYPE_UINT},
 123
 124   {"TX_CQ_LEN", "4096",
 125    "Length of send completion queue. This limits the total number of outstanding signaled sends.",
 126    ucs_offsetof(uct_rc_iface_config_t, tx_cq_len), UCS_CONFIG_TYPE_UINT},
 127

*/

typedef enum {
  NCCL_UCT_START = 0,
  NCCL_UCT_CONNECT,
  NCCL_UCT_ACCEPT,
  NCCL_UCT_RECEIVE_COMM,
  NCCL_UCT_RECEIVE_ADDR,
  NCCL_UCT_RX_READY,
  NCCL_UCT_DONE
} nccl_uct_state_t;

typedef enum {
  NCCL_UCT_AM_RTR = 14, /* Use particular values */
  NCCL_UCT_AM_ATP = 15,
  NCCL_UCT_AM_RTS = 16,
  NCCL_UCT_AM_ATS = 17
} nccl_uct_am_type_t;

typedef enum {
  NCCL_UCT_REQ_IRECV  = -1,
  NCCL_UCT_REQ_IFLUSH = -2
} nccl_uct_request_type_t;

/* UCT EP address to exchange and connect to */
typedef struct {
  uint8_t dev_addr_size;
  uint8_t ep_addr_size;
  uint8_t data[64];
} nccl_uct_ep_addr_t;

typedef struct {
  uct_iface_h     iface;
  uct_md_h        md;
  uct_component_h comp;
  void            *addr;
  size_t          addr_size;
  void            *dev_addr;
  size_t          dev_addr_size;
  size_t          ep_addr_size;
  size_t          rkey_packed_size;

  size_t          am_max_short;
} nccl_uct_iface_t;

struct nccl_uct_context;

typedef struct nccl_uct_worker {
  struct nccl_uct_worker *next;
  struct {
    pthread_t thread;
    int       dev;
  } id;

  int                     count;
  ucs_async_context_t     *async;
  uct_worker_h            worker;
  nccl_uct_iface_t        *uct_iface;
  struct nccl_uct_context *context;
} nccl_uct_worker_t;

typedef struct {
  uct_ep_h         ep;
  uct_ep_addr_t    *addr;
  size_t           addr_size;
  nccl_uct_iface_t *uct_iface;
  uint8_t          data[];
} nccl_uct_ep_t;

struct nccl_uct_rdesc;
struct nccl_uct_comm;

/* On-the-wire descriptor of a posted receive request entry */
typedef struct {
  int        tag;
  int        size;
  void       *data;
  int        matched;
  uct_rkey_t rkey;
} nccl_uct_chunk_t;

/* On-the-wire descriptor of a receive request containing many chunks */
typedef struct {
  uint64_t              id;
  uint16_t              count;
  uint32_t              size;
  struct nccl_uct_rdesc *peer_rdesc; /* Acts as a cookie along with id */
  nccl_uct_chunk_t      chunk[];
} nccl_uct_rdesc_hdr_t;

/* On-the-wire descriptor for receive request completion */
typedef struct {
  uint64_t              id;
  struct nccl_uct_rdesc *rdesc;
  int                   count; /* Number of sizes contained */
  int                   sizes[NCCL_UCX_UCT_MAX_RECVS];
} nccl_uct_atp_t;

/* TODO add comments */
/* TODO merge rts with other chunk */

/* Memory registration handle in NCCL UCT plugin returned by ->regMR() */
typedef struct {
  uct_mem_h            memh;
  struct nccl_uct_comm *comm;
  uct_rkey_bundle_t    bundle;
  uint8_t              rkey[];
} nccl_uct_memh_t;

typedef struct {
  int        tag;
  int        size;
  void       *data;
  union {
    uct_rkey_t      rkey;
    nccl_uct_memh_t *uct_memh;
  } u;
  struct nccl_uct_rd_req *opaque;
  unsigned index;
} nccl_uct_mem_t;

/*
 * NCCL local request handler to progress:
 * - size -1 for multi receive
 * - size -2 for flush
 * - size > 0 for send
 */
typedef struct {
  /* Pending GET (iflush) PUT (isend) or receiving one ATP (irecv) */
  uct_completion_t      completion;
  int                   size;
  struct nccl_uct_rdesc *rdesc;
} nccl_uct_req_t;

/* Pending receive descriptor either on the receive or sending side */
typedef struct nccl_uct_rdesc {
  int                   nccl_usage; /* NCCL requests not finished/started */
  int                   send_atp;   /* >1 pending isend, ==1 pending atp send */

  struct nccl_uct_rdesc *prev; /* comm's linked list */
  struct nccl_uct_rdesc *next;

  struct nccl_uct_comm  *comm;
  nccl_uct_rdesc_hdr_t  desc;
  nccl_uct_chunk_t      storage[NCCL_UCX_UCT_MAX_RECVS]; /* Don't use directly */
  nccl_uct_req_t        reqs[NCCL_UCX_UCT_MAX_RECVS];    /* NCCL requests */
  int                   sizes[NCCL_UCX_UCT_MAX_RECVS];   /* ATP received sizes */
} nccl_uct_rdesc_t;

/* All the remote addresses for the communicator */
typedef struct nccl_uct_comm_addr {
  nccl_uct_ep_addr_t rma;
  nccl_uct_ep_addr_t am;
  /* TODO: Add multi-QP here */
} nccl_uct_comm_addr_t;

/* TODO: static assert wwrt MAX REQUESTS */
#define NCCL_UCT_RING_SIZE      (1 << 10)
#define NCCL_UCT_RING_MASK      (NCCL_UCT_RING_SIZE - 1)

#define NCCL_UCT_RING_HASH_SIZE (1 << 4)  /* 16 */
#define NCCL_UCT_RING_HASH_MASK (NCCL_UCT_RING_HASH_SIZE - 1)

typedef struct nccl_uct_ring {
  unsigned            first;
  unsigned            last;

  /* Key */
  int                 tag[NCCL_UCT_RING_SIZE];

  /* Value */
  struct nccl_uct_ring_data {
    nccl_uct_mem_t mem;
  } entry[NCCL_UCT_RING_SIZE];
} nccl_uct_ring_t;

static void nccl_uct_ring_init(nccl_uct_ring_t *ring)
{
  int i;

  ring->first = 0;
  ring->last  = 0;
  for (i = 0; i < NCCL_UCT_RING_SIZE; i++) {
    ring->tag[i] = UINT_MAX;
  }
}

static inline unsigned nccl_uct_ring_hash(int tag)
{
  unsigned v = tag;
  v ^= v >> 1;
  return v & NCCL_UCT_RING_HASH_MASK;
}

static inline void nccl_uct_ring_append(nccl_uct_ring_t *ring, int tag,
                                        void *data, size_t len)
{
  int j = ring->last & NCCL_UCT_RING_MASK;

  ring->last++;

  assert(ring->tag[j] == UINT_MAX);
  assert(len <= sizeof(ring->entry[0]));

  /* Touch two cache-lines */
  ring->tag[j] = tag;
  memcpy(&ring->entry[j].mem, data, len);

  assert((ring->last & NCCL_UCT_RING_MASK) !=
         (ring->first & NCCL_UCT_RING_MASK));
}

static inline void nccl_uct_ring_consume(nccl_uct_ring_t *ring, unsigned i)
{
  unsigned j = i & NCCL_UCT_RING_MASK;

  assert(ring->tag[j] != UINT_MAX);
  ring->tag[j] = UINT_MAX;

  /* Cleanup upon tag hit */
  if (i == ring->first) {
    for (; i != ring->last; i++) {
      j = i & NCCL_UCT_RING_MASK;
      if (ring->tag[j] != UINT_MAX) {
        break;
      }
      ring->first = i + 1;
    }
  }
}

static inline nccl_uct_mem_t *
nccl_uct_ring_get_mem(nccl_uct_ring_t *ring, unsigned i)
{
  return &ring->entry[i & NCCL_UCT_RING_MASK].mem;
}

static inline unsigned nccl_uct_ring_find(nccl_uct_ring_t *ring, int tag)
{
  int i;

  assert(tag != UINT_MAX);

  for (i = ring->first; i != ring->last; i++) {
    if (ring->tag[i & NCCL_UCT_RING_MASK] == tag) {
      return i;
    }
  }

  return ring->last;
}

typedef struct {
  uct_iov_t              iov;
  uint64_t               rva;
  uct_rkey_t             rkey;
  uct_completion_t       *completion;
  struct nccl_uct_rd_req *req;
} nccl_uct_get_param_t;

#define NCCL_UCT_GET_PARAM_SIZE 512
#define NCCL_UCT_GET_PARAM_MASK (NCCL_UCT_GET_PARAM_SIZE - 1)

/* Either Receiver or Sender communicator, connected to one peer */
typedef struct nccl_uct_comm {
  struct ncclSocket       sock;
  struct nccl_uct_context *context;
  int                     dev;

  nccl_uct_worker_t       *uct_worker;
  nccl_uct_iface_t        *uct_iface;
  nccl_uct_ep_t           *uct_ep;
  nccl_uct_ep_t           *uct_ep_am;

  nccl_uct_comm_addr_t    addr; /* Remote addresses */

  int                     rdesc_alloc; /* Track allocated rdescs */
  nccl_uct_rdesc_t        *free_rdesc; /* Available rdesc for reuse */
  uint64_t                rdesc_id;    /* Next sequence number to use */

  /* Read mode section */
  int                     req_count;
  struct nccl_uct_rd_req  *free_req;
  nccl_uct_ring_t         exp[NCCL_UCT_RING_HASH_SIZE];
  nccl_uct_ring_t         unexp[NCCL_UCT_RING_HASH_SIZE];

  /* Get zcopy yet to be started */
  struct {
    unsigned                first;
    unsigned                last;
    nccl_uct_get_param_t    param[NCCL_UCT_GET_PARAM_SIZE];
  } get;


  const struct nccl_uct_comm *remote_comm; /* Cookie received while connecting */

  /* Local GET on current device */
  struct {
    int                enabled;
    nccl_uct_ep_t      *uct_ep; /* Locally read from HCA */
    nccl_uct_ep_addr_t addr;

    uint8_t            mem[65]; /* Dummy memory to read into */
    uct_mem_h          memh;
  } gpu_flush;

  /* Received RTRs: used by Sender communicator in ->isend() */
  struct {
    nccl_uct_rdesc_t *head;
    nccl_uct_rdesc_t *tail;
  } rdesc_list;
} nccl_uct_comm_t;

/* State tracking used while connecting/accepting only */
typedef struct {
  nccl_uct_state_t state;
  nccl_uct_comm_t  *comm;  /* current communicator being created */
  int              offset; /* for Socket reading */
  int              ready;  /* accept must complete after connect */
} nccl_uct_stage_t;

/* On-the-wire handle passed OOB by NCCL from listener to connector */
typedef struct {
  uint64_t                  magic;
  struct {
    union ncclSocketAddress addr;
    uint32_t                id;
  } listener;
  nccl_uct_comm_t           *comm; /* Created communicator in accept */
  nccl_uct_stage_t          stage; /* Used by connector */
} nccl_uct_listen_handle_t;

/* Communicator while listening to remote ranks */
typedef struct {
  struct ncclSocket       sock;
  struct nccl_uct_context *context;
  int                     dev;
  uint32_t                id;
  nccl_uct_worker_t       *uct_worker;
  nccl_uct_comm_t         *comm;

  /* Used by acceptor */
  nccl_uct_stage_t        stage;
} nccl_uct_listen_comm_t;

/* Global state of the plugin */
typedef struct nccl_uct_context {
  /* Transport to use */
  const char              *tl_name;

  /* IB devices available */
  int                     dev_count;

  /* OOB socket for accepting/connecting */
  char                    if_name[MAX_IF_NAME_SIZE];
  union ncclSocketAddress if_addr;

  /* Number of listener created */
  uint32_t                listener_count;

  /* List of created workers */
  nccl_uct_worker_t       *worker_list;
} nccl_uct_context_t;

/* Either from read or write side */
typedef struct nccl_uct_rd_req {
  uct_completion_t        completion; /* Keep first */
  int                     send_rts;
  int                     count;
  int                     rts_count; /* RTS received */
  int                     started;
  int                     ats_sent;

  struct nccl_uct_rd_req  *remote_req[NCCL_UCX_UCT_MAX_RECVS];

  struct nccl_uct_rd_req  *next;
  nccl_uct_comm_t         *comm;

  nccl_uct_mem_t          rts;
  nccl_uct_mem_t          recv[NCCL_UCX_UCT_MAX_RECVS];
} nccl_uct_rd_req_t;

static pthread_mutex_t nccl_uct_lock = PTHREAD_MUTEX_INITIALIZER;

static nccl_uct_context_t context = {
    .tl_name = "rc_mlx5",
    .dev_count = -1
};

static const uct_device_addr_t *
nccl_uct_ep_addr_dev(const nccl_uct_ep_addr_t *addr) {
  return (uct_device_addr_t*)addr->data;
}

static const uct_ep_addr_t *
nccl_uct_ep_addr_ep(const nccl_uct_ep_addr_t *addr) {
  return (uct_ep_addr_t*)(addr->data + addr->dev_addr_size);
}

static ncclResult_t nccl_uct_ep_addr_set(nccl_uct_ep_addr_t *addr,
                                         const nccl_uct_comm_t *comm,
                                         const nccl_uct_ep_t *uct_ep) {
  nccl_uct_iface_t *uct_iface = comm->uct_iface;
  size_t total = uct_iface->dev_addr_size + uct_iface->ep_addr_size;

  if (total > sizeof(addr->data)) {
    WARN("Address sizes are too big (%zu + %u > %zu)", uct_iface->dev_addr_size,
         uct_iface->ep_addr_size);
    return ncclSystemError;
  }

  addr->dev_addr_size = uct_iface->dev_addr_size;
  addr->ep_addr_size  = uct_iface->ep_addr_size;

  memcpy(addr->data, uct_iface->dev_addr, addr->dev_addr_size);
  memcpy(addr->data + addr->dev_addr_size, uct_ep->addr,
         uct_iface->ep_addr_size);
  return ncclSuccess;
}

static uct_iface_h nccl_uct_resource_iface_open(uct_worker_h worker,
                                                uct_md_h md,
                                                uct_tl_resource_desc_t *tl) {
  uct_iface_params_t params = {};
  ucs_status_t status;
  uct_iface_config_t *config;
  uct_iface_h iface;

  status = uct_md_iface_config_read(md, tl->tl_name, NULL, NULL, &config);
  if (status != UCS_OK) {
    WARN("Failed to read MD iface config for TL '%s': error %d", tl->tl_name,
         status);
    return NULL;
  }

  status = uct_config_modify(config, "RC_MLX5_TX_CQ_MODERATION", "128");
  if (status != UCS_OK) {
    WARN("Failed to modify MD iface config CQ MOD for TL '%s': error %d", tl->tl_name, status);
		return NULL;
  }

  status = uct_config_modify(config, "RC_MLX5_TX_CQ_LEN", "8192");
  if (status != UCS_OK) {
    WARN("Failed to modify MD iface config CQ LEN for TL '%s': error %d", tl->tl_name, status);
	  return NULL;
	}


  params.field_mask =
      UCT_IFACE_PARAM_FIELD_OPEN_MODE | UCT_IFACE_PARAM_FIELD_DEVICE |
      UCT_IFACE_PARAM_FIELD_STATS_ROOT | UCT_IFACE_PARAM_FIELD_RX_HEADROOM;
  params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
  params.mode.device.tl_name  = tl->tl_name;
  params.mode.device.dev_name = tl->dev_name;
  params.stats_root           = NULL;
  params.rx_headroom          = 0;

  status = uct_iface_open(md, worker, &params, config, &iface);
  uct_config_release(config);
  if (status != UCS_OK) {
    WARN("Failed to open iface %s/%s: error %d", tl->tl_name, tl->dev_name,
         status);
    return NULL;
  }

  uct_iface_progress_enable(iface, UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

  return iface;
}

static uct_iface_h
nccl_uct_md_iface_open(uct_worker_h worker, uct_component_h comp,
                       unsigned md_index, const char *md_name,
                       const char *tl_name, const char *dev_name,
                       uct_md_h *md_p) {
  uct_iface_h iface = NULL;
  ucs_status_t status;
  uct_md_config_t *md_config;
  uct_md_h md;
  uct_md_attr_t md_attr;
  uct_tl_resource_desc_t *tls;
  unsigned tls_count, i;

  status = uct_md_config_read(comp, NULL, NULL, &md_config);
  if (status != UCS_OK) {
    WARN("Failed to read MD[%d] config: error %d", md_index, status);
    return NULL;
  }

  status = uct_md_open(comp, md_name, md_config, &md);
  uct_config_release(md_config);
  if (status != UCS_OK) {
    WARN("Failed to open MD[%d/%s]: error %d", md_index, md_name, status);
    return NULL;
  }

  status = uct_md_query(md, &md_attr);
  if (status != UCS_OK) {
    WARN("Failed to query MD[%d/%s]: error %d", md_index, md_name, status);
    goto out;
  }

  status = uct_md_query_tl_resources(md, &tls, &tls_count);
  if (status != UCS_OK) {
    WARN("Failed to query resources MD[%d/%s]; error %d", md_index, md_name,
         status);
    goto out;
  }

  for (i = 0; i < tls_count; i++) {
    if (!strcmp(dev_name, tls[i].dev_name) &&
        !strcmp(tl_name, tls[i].tl_name)) {

      iface = nccl_uct_resource_iface_open(worker, md, &tls[i]);
      break;
    }
  }

  uct_release_tl_resource_list(tls);

out:
  if (iface == NULL) {
    uct_md_close(md);
  } else {
    *md_p = md;
  }
  return iface;
}

static void nccl_uct_ep_destroy(nccl_uct_ep_t *uct_ep) {
  uct_ep_destroy(uct_ep->ep);
  free(uct_ep);
}

static nccl_uct_ep_t *nccl_uct_ep_create(nccl_uct_iface_t *uct_iface) {
  nccl_uct_ep_t *uct_ep;
  ucs_status_t status;
  uct_ep_params_t ep_params;

  uct_ep = calloc(1, sizeof(*uct_ep) + uct_iface->ep_addr_size);
  if (uct_ep == NULL) {
    WARN("Failed to alloc EP memory");
    return NULL;
  }

  uct_ep->addr      = (uct_ep_addr_t*)uct_ep->data;
  uct_ep->uct_iface = uct_iface;

  ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
  ep_params.iface      = uct_iface->iface;

  status = uct_ep_create(&ep_params, &uct_ep->ep);
  if (status != UCS_OK) {
    WARN("Failed to create UCT EP: error %d", status);
    free(uct_ep);
    return NULL;
  }

  status = uct_ep_get_address(uct_ep->ep, uct_ep->addr);
  if (status != UCS_OK) {
    WARN("Failed to get UCT EP address: error %d", status);
    nccl_uct_ep_destroy(uct_ep);
    return NULL;
  }

  return uct_ep;
}

static ncclResult_t nccl_uct_ep_connect_to_ep(nccl_uct_ep_t *uct_ep,
                                              nccl_uct_ep_addr_t *addr) {
  ucs_status_t status = uct_ep_connect_to_ep(
      uct_ep->ep, nccl_uct_ep_addr_dev(addr), nccl_uct_ep_addr_ep(addr));
  if (status != UCS_OK) {
    WARN("Failed to connect to EP: error %d", status);
    return ncclSystemError;
  }

  return ncclSuccess;
}

static void nccl_uct_comm_rdesc_insert(nccl_uct_rdesc_t *rdesc) {
  nccl_uct_comm_t *comm = rdesc->comm;

  if (comm->rdesc_list.tail != NULL) {
    rdesc->prev                 = comm->rdesc_list.tail;
    comm->rdesc_list.tail->next = rdesc;
  } else {
    assert(comm->rdesc_list.head == NULL);
    rdesc->prev           = NULL;
    comm->rdesc_list.head = rdesc;
  }

  comm->rdesc_list.tail = rdesc;
}

static void nccl_uct_comm_rdesc_del(nccl_uct_rdesc_t *rdesc) {
  nccl_uct_comm_t *comm = rdesc->comm;

  if (rdesc->prev != NULL) {
    rdesc->prev->next = rdesc->next;
  }
  if (rdesc->next != NULL) {
    rdesc->next->prev = rdesc->prev;
  }
  if (rdesc == comm->rdesc_list.head) {
    comm->rdesc_list.head = rdesc->next;
  }
  if (rdesc == comm->rdesc_list.tail) {
    comm->rdesc_list.tail = rdesc->prev;
  }
}

static nccl_uct_rdesc_t *nccl_uct_comm_rdesc_get(nccl_uct_comm_t *comm) {
  nccl_uct_rdesc_t *rdesc = comm->free_rdesc;

  if (rdesc == NULL) {
    rdesc = calloc(1, sizeof(*rdesc));
  } else {
    comm->free_rdesc = rdesc->next;
  }

  rdesc->next = NULL;
  rdesc->comm = comm;
  comm->rdesc_alloc++;
  return rdesc;
}

static size_t nccl_uct_rdesc_size(int n) {
  return n * sizeof(nccl_uct_chunk_t) + sizeof(nccl_uct_rdesc_hdr_t);
}

/* Prepare a receive descriptor from irecv()/iflush() side */
static void nccl_uct_rdesc_set(nccl_uct_rdesc_t *rdesc, uint64_t id, int n,
                               void **data, int *sizes, int *tags,
                               nccl_uct_memh_t **uct_memh) {
  nccl_uct_rdesc_hdr_t *desc = &rdesc->desc;
  int i;

  /* Populate header */
  desc->id         = id;
  desc->count      = n;
  desc->size       = nccl_uct_rdesc_size(n);
  desc->peer_rdesc = rdesc; /* cookie, will be returned in ATP */

  /* Ref count that prevents NCCL from releasing memory */
  rdesc->nccl_usage = 1;
  rdesc->send_atp   = 0;

  /* Zero (iflush) or One or many receive request are contained */
  for (i = 0; i < n; i++) {
    desc->chunk[i].tag     = tags[i];
    desc->chunk[i].size    = sizes[i];
    desc->chunk[i].data    = data[i];
    desc->chunk[i].matched = 0;
    desc->chunk[i].rkey    = uct_memh[i]->bundle.rkey;
  }
}

static void nccl_uct_empty_callback(uct_completion_t *comp) {
}

static nccl_uct_req_t *nccl_uct_rdesc_get_req(nccl_uct_rdesc_t *rdesc, int i,
                                              int size) {
  nccl_uct_req_t *req;

  assert(i < NCCL_UCX_UCT_MAX_RECVS);

  req        = &rdesc->reqs[i];
  req->size  = size;
  req->rdesc = rdesc;

  req->completion.func   = nccl_uct_empty_callback;
  req->completion.count  = 1;
  req->completion.status = UCS_OK;

  return &rdesc->reqs[i];
}

static void nccl_uct_comm_rdesc_put(nccl_uct_rdesc_t *rdesc) {
  nccl_uct_comm_t *comm = rdesc->comm;

  assert(comm != NULL);

  rdesc->desc.id   = -1;
  rdesc->comm      = NULL;
  rdesc->next      = comm->free_rdesc;
  comm->free_rdesc = rdesc;
  comm->rdesc_alloc--;
}

/* On receiver side, after ->irecv(), expect corresponding ATP */
static ucs_status_t nccl_uct_atp_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
  nccl_uct_atp_t *atp = (nccl_uct_atp_t*)((uint8_t*)data + 8);

  assert(length == (sizeof(*atp) + 8));
  assert(*(nccl_uct_comm_t**)data == atp->rdesc->comm);
  assert(atp->id == atp->rdesc->desc.id);
  assert(atp->count == atp->rdesc->desc.count);
  assert(atp->rdesc->reqs[0].completion.count == 1);

  atp->rdesc->reqs[0].completion.count--;
  memcpy(atp->rdesc->sizes, atp->sizes, atp->count * sizeof(*atp->sizes));
  return UCS_OK;
}

/* On sender side, asynchronously receive rdesc/RTR, later used by ->isend() */
static ucs_status_t nccl_uct_rtr_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
  nccl_uct_comm_t *comm      = *(nccl_uct_comm_t**)data;
  nccl_uct_rdesc_hdr_t *desc = (nccl_uct_rdesc_hdr_t*)((uint8_t*)data + 8);
  size_t size                = desc->size;
  nccl_uct_rdesc_t *rdesc;

  rdesc = nccl_uct_comm_rdesc_get(comm);
  if (rdesc == NULL) {
    WARN("Failed to get an rdesc in RTR callback");
    return UCS_ERR_NO_MEMORY; /* Cannot happend */
  }

  nccl_uct_comm_rdesc_insert(rdesc); /* ->isend() will search for tags */

  assert((size + 8) == length);
  assert(size == nccl_uct_rdesc_size(desc->count));

  memcpy(&rdesc->desc, desc, size);
  rdesc->nccl_usage = desc->count;
  rdesc->send_atp   = desc->count + 1;
  return UCS_OK;
}

static void nccl_uct_rd_send_ats(nccl_uct_rd_req_t *req)
{
  ucs_status_t status;

  assert(req->ats_sent == 0);
  assert(req->send_rts == 0);
  assert(req->rts_count == req->count);
  assert(req->completion.count == 1);

  status = uct_ep_fence(req->comm->uct_ep->ep, 0);
  assert(status == UCS_OK);

  status = uct_ep_am_short(req->comm->uct_ep->ep, NCCL_UCT_AM_ATS,
                           (uint64_t)req->comm->remote_comm,
                           req->remote_req,
                           sizeof(*req->remote_req) * req->rts_count);
  if (status == UCS_OK) {
    req->ats_sent = 1;
    req->completion.count--;
  }
}

static ucs_status_t nccl_uct_get_zcopy(nccl_uct_comm_t *comm,
                                       nccl_uct_get_param_t *param)
{
  ucs_status_t status;

  if (param->iov.length == 0) {
    param->completion->count--;
		goto done;
  }

  status = uct_ep_get_zcopy(comm->uct_ep->ep, &param->iov, 1, param->rva,
                            param->rkey, param->completion);
  if (status == UCS_OK) {
    param->completion->count--;
  } else if (status != UCS_INPROGRESS) {
    return status;
  }

done:
  param->req->started++;
  assert(param->req->started <= param->req->count);

  if (param->completion->count == 1) {
    nccl_uct_rd_send_ats(param->req);
  }

  return UCS_OK;
}

static void nccl_uct_match_add(nccl_uct_comm_t *comm, nccl_uct_mem_t *src,
                               nccl_uct_mem_t *dst)
{
  nccl_uct_rd_req_t *req = dst->opaque;
  nccl_uct_get_param_t *param;
  int i;

  i     = comm->get.last++;
  i    &= NCCL_UCT_GET_PARAM_MASK;
  param = &comm->get.param[i];

  assert(src->size <= dst->size);
  assert((comm->get.first & NCCL_UCT_GET_PARAM_MASK) !=
         (comm->get.last & NCCL_UCT_GET_PARAM_MASK));

  dst->size = src->size;
	req->recv[dst->index].size = src->size;

  req->remote_req[req->rts_count++] = src->opaque;
  assert(req->rts_count <= NCCL_UCX_UCT_MAX_RECVS);

  param->iov.buffer = dst->data;
  param->iov.length = dst->size;
  param->iov.memh   = dst->u.uct_memh->memh;
  param->iov.stride = 0;
  param->iov.count  = 1;
  param->rva        = (uint64_t)src->data;
  param->rkey       = src->u.rkey;
  param->completion = &req->completion;
  param->req        = req;
}

static inline void nccl_uct_match_drain(nccl_uct_comm_t *comm)
{
  int i;
  ucs_status_t status;

  for (; comm->get.first != comm->get.last; comm->get.first++) {
    i = comm->get.first & NCCL_UCT_GET_PARAM_MASK;

    status = nccl_uct_get_zcopy(comm, &comm->get.param[i]);
    if (status != UCS_OK) {
      break;
    }
  }
}

static nccl_uct_rd_req_t *nccl_uct_rd_req_alloc(nccl_uct_comm_t *comm)
{
  nccl_uct_rd_req_t *req = comm->free_req;

  if (req == NULL) {
    req = malloc(sizeof(*req));
    if (req == NULL) {
      return req;
    }
  } else {
    comm->free_req = req->next;
  }

  req->comm = comm;
  req->comm->req_count++;
  return req;
}

static inline void nccl_uct_rd_req_free(nccl_uct_rd_req_t *req)
{
  req->next           = req->comm->free_req;
  req->comm->free_req = req;
  req->comm->req_count--;
}

ncclResult_t nccl_uct_rd_irecv(void *recv_comm, int n, void **data,
                               int *sizes, int *tags, void **mhandles,
                               void **request) {
  nccl_uct_comm_t *comm      = recv_comm;
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandles;
  nccl_uct_ring_t *unexp;
  nccl_uct_rd_req_t *req;
  nccl_uct_mem_t *rts;
  int i, j, index;

  /* Create a request */
  req = nccl_uct_rd_req_alloc(comm);
  *request = req;
  if (req == NULL) {
    return ncclSuccess;
  }

  req->send_rts           = 0;
  req->completion.func    = nccl_uct_empty_callback;
  req->completion.count   = n + 1;
  req->completion.status  = UCS_OK;
  req->count              = n;
  req->rts_count          = 0;
  req->started            = 0;
  req->ats_sent           = 0;

  assert(n <= NCCL_UCX_UCT_MAX_RECVS);

  for (i = 0; i < n; i++) {
    req->recv[i].tag        = tags[i];
    req->recv[i].size       = sizes[i];
    req->recv[i].data       = data[i];
    req->recv[i].u.uct_memh = uct_memh[i];
    req->recv[i].opaque     = req;
    req->recv[i].index      = i;
  }

  /* Build match list */
  for (i = 0; i < n; i++) {
    index = nccl_uct_ring_hash(tags[i]);
    unexp = &comm->unexp[index];
    j     = nccl_uct_ring_find(unexp, tags[i]);
    if (j == unexp->last) {
      nccl_uct_ring_append(&comm->exp[index], tags[i], &req->recv[i],
                           sizeof(req->recv[i]));
    } else {
      rts = nccl_uct_ring_get_mem(unexp, j);
      nccl_uct_match_add(comm, rts, &req->recv[i]);
      nccl_uct_ring_consume(unexp, j);
    }
  }

  nccl_uct_match_drain(comm);
  return ncclSuccess;
}

static ucs_status_t nccl_uct_ats_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
  nccl_uct_comm_t *comm = *(nccl_uct_comm_t**)data;
  nccl_uct_rd_req_t **req = (nccl_uct_rd_req_t **)((uint8_t *)data + 8);
  nccl_uct_rd_req_t **end  = (nccl_uct_rd_req_t **)((uint8_t *)data + length);


  (void)comm;
  for (; req + 1 <= end; req++) {
    assert((*req)->completion.count == 1);
    assert((*req)->comm == comm);

    (*req)->completion.count = 0;

  }
  return UCS_OK;
}

/* TODO: Pro: one lookup at irecv post or one lookup at isend recv */
static ucs_status_t nccl_uct_rts_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
  nccl_uct_comm_t *comm = *(nccl_uct_comm_t**)data;
  nccl_uct_mem_t *rts   = (nccl_uct_mem_t *)((uint8_t *)data + 8);
  nccl_uct_ring_t *exp;
  nccl_uct_mem_t *dst;
  int i, index;

  assert(length == (sizeof(*rts) + 8));

  /* Do we already expect it? */
  index = nccl_uct_ring_hash(rts->tag);
  exp   = &comm->exp[index];
  i     = nccl_uct_ring_find(exp, rts->tag);
  if (i == exp->last) {
    nccl_uct_ring_append(&comm->unexp[index], rts->tag, rts, sizeof(*rts));
  } else {
    /* Receive request was already posted */
    dst = nccl_uct_ring_get_mem(exp, i);
    nccl_uct_match_add(comm, rts, dst);
    nccl_uct_ring_consume(exp, i);
  }

  return UCS_OK;
}

static ncclResult_t nccl_uct_iface_set_handler(nccl_uct_iface_t *uct_iface,
                                               int id,
                                               uct_am_callback_t callback) {
  ucs_status_t status =
      uct_iface_set_am_handler(uct_iface->iface, id, callback, NULL, 0);
  if (status != UCS_OK) {
    WARN("Failed to get AM handler id=%d status=%d", id, status);
    return ncclInternalError;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_iface_set_rtr_mode(nccl_uct_iface_t *uct_iface,
                                                nccl_uct_comm_t *comm) {
  NCCLCHECK(nccl_uct_iface_set_handler(uct_iface, NCCL_UCT_AM_RTR,
                                       nccl_uct_rtr_callback));
  NCCLCHECK(nccl_uct_iface_set_handler(uct_iface, NCCL_UCT_AM_ATP,
                                       nccl_uct_atp_callback));
  return ncclSuccess;
}

static void nccl_uct_iface_close(nccl_uct_iface_t *uct_iface) {
  uct_iface_close(uct_iface->iface);
  uct_md_close(uct_iface->md);
  free(uct_iface->dev_addr);
  free(uct_iface->addr);
  free(uct_iface);
}

static nccl_uct_iface_t *nccl_uct_iface_open(nccl_uct_worker_t *uct_worker,
                                             const char *tl_name,
                                             const char *dev_name) {
  uct_worker_h worker         = uct_worker->worker;
  nccl_uct_iface_t *uct_iface = NULL;
  uct_iface_h iface           = NULL;
  uct_component_h *comps, *comp;
  unsigned comps_count, i;
  ucs_status_t status;
  uct_component_attr_t comp_attr;
  uct_iface_attr_t iface_attr;
  uct_md_h md;
  uct_md_attr_t md_attr;
  int rtr_size;

  status = uct_query_components(&comps, &comps_count);
  if (status != UCS_OK) {
    WARN("Failed to query component list: error %d", status);
    return NULL;
  }

  for (comp = comps; comp < comps + comps_count; comp++) {
    comp_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT;
    status               = uct_component_query(*comp, &comp_attr);
    if (status != UCS_OK) {
      WARN("Failed to query component: error %d", status);
      goto out;
    }

    comp_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    comp_attr.md_resources =
        alloca(sizeof(*comp_attr.md_resources) * comp_attr.md_resource_count);
    status = uct_component_query(*comp, &comp_attr);
    if (status != UCS_OK) {
      WARN("Failed to query component resources: error %d", status);
      goto out;
    }

    for (i = 0; i < comp_attr.md_resource_count; i++) {
      iface = nccl_uct_md_iface_open(worker, *comp, i,
                                     comp_attr.md_resources[i].md_name, tl_name,
                                     dev_name, &md);
      if (iface != NULL) {
        goto found;
      }
    }
  }

  if (iface == NULL) {
    WARN("Failed to open iface for tl_name=%s dev_name=%s", tl_name, dev_name);
    goto out;
  }

found:
  status = uct_iface_query(iface, &iface_attr);
  if (status != UCS_OK) {
    WARN("Failed to query iface for tl_name=%s dev_name=%s", tl_name, dev_name);
    goto fail;
  }

  if (!(iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP)) {
    WARN("Interface flag CONNECT_TO_EP is not set");
    goto fail;
  }

  if (!(iface_attr.cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) ||
      !(iface_attr.cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY)) {
    WARN("Interface does not support ZCOPY (flags=0x%x)", iface_attr.cap.flags);
    goto fail;
  }

  status = uct_md_query(md, &md_attr);
  if (status != UCS_OK) {
    WARN("Failed to query md for iface %p: error %d", iface, status);
    goto fail;
  }

  uct_iface = calloc(1, sizeof(*uct_iface));
  if (uct_iface == NULL) {
    WARN("Failed to alloc uct iface structure");
    goto fail;
  }

  uct_iface->ep_addr_size     = iface_attr.ep_addr_len;
  uct_iface->md               = md;
  uct_iface->comp             = *comp;
  uct_iface->rkey_packed_size = md_attr.rkey_packed_size;

  if (iface_attr.cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
    uct_iface->am_max_short = iface_attr.cap.am.max_short;
  }

  rtr_size = nccl_uct_rdesc_size(NCCL_UCX_UCT_MAX_RECVS);
  if (rtr_size > uct_iface->am_max_short) {
    WARN("Failed RTR does not fit in face AM short (%dB > %dB)", rtr_size,
         uct_iface->am_max_short);
    goto fail;
  }

  if (uct_iface->rkey_packed_size > sizeof(((nccl_uct_chunk_t*)0)->rkey)) {
    WARN("Interface rkey_packed_size %d too big", uct_iface->rkey_packed_size);
    goto fail;
  }

  if (iface_attr.device_addr_len > 0) {
    uct_iface->dev_addr_size = iface_attr.device_addr_len;
    uct_iface->dev_addr      = calloc(1, iface_attr.device_addr_len);
    if (uct_iface->dev_addr == NULL) {
      WARN("Failed to alloc dev_addr");
      goto fail;
    }

    status = uct_iface_get_device_address(iface, uct_iface->dev_addr);
    if (status != UCS_OK) {
      WARN("Failed to query iface device addr for tl_name=%s dev_name=%s",
           tl_name, dev_name);
      goto fail;
    }
  }

  if (iface_attr.iface_addr_len > 0) {
    uct_iface->addr_size = iface_attr.iface_addr_len;
    uct_iface->addr      = calloc(1, iface_attr.iface_addr_len);
    if (uct_iface->addr == NULL) {
      WARN("Failed to alloc iface addr");
      goto fail;
    }

    status = uct_iface_get_address(iface, uct_iface->addr);
    if (status != UCS_OK) {
      WARN("Failed to query iface addr to tl_name=%s dev_name=%s", tl_name,
           dev_name);
      goto fail;
    }
  }

  uct_iface->iface = iface;

out:
  uct_release_component_list(comps);
  return uct_iface;

fail:
  if (uct_iface != NULL) {
    free(uct_iface->dev_addr);
    free(uct_iface->addr);
    free(uct_iface);
  }
  if (iface != NULL) {
    uct_iface_close(iface);
  }
  uct_release_component_list(comps);
  return NULL;
}

static ncclResult_t nccl_uct_init(ncclDebugLogger_t logFunction) {
  return nccl_p2p_ib_init(&context.dev_count, ncclIbDevs, context.if_name,
                          &context.if_addr, NULL, logFunction);
}

static ncclResult_t nccl_uct_devices(int *ndev) {
  *ndev = context.dev_count;
  return ncclSuccess;
}

static ncclResult_t nccl_uct_get_properties(int dev,
                                            ncclNetProperties_t *props) {
  return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

static const char *nccl_dev_name(int dev) {
  static __thread char buf[128];
  snprintf(buf, sizeof(buf), "%s:%d", ncclIbDevs[dev].devName,
           ncclIbDevs[dev].portNum);
  return buf;
}

static ncclResult_t nccl_uct_worker_create(nccl_uct_worker_t *w,
                                           nccl_uct_context_t *context,
                                           int dev) {
  ucs_status_t status;

  status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK, &w->async);
  if (status != UCS_OK) {
    WARN("Failed to create UCT async context: dev=%d", dev);
    goto fail;
  }

  status = uct_worker_create(w->async, UCS_THREAD_MODE_SINGLE, &w->worker);
  if (status != UCS_OK) {
    WARN("Failed to create UCT worker: dev=%d", dev);
    ucs_async_context_destroy(w->async);
    goto fail;
  }

  w->id.dev    = dev;
  w->id.thread = pthread_self();
  w->context   = context;

  w->uct_iface = nccl_uct_iface_open(w, context->tl_name, nccl_dev_name(dev));
  if (w->uct_iface == NULL) {
    WARN("Failed to create UCT iface for worker: dev=%d", dev);
    uct_worker_destroy(w->worker);
    ucs_async_context_destroy(w->async);
    goto fail;
  }

  NCCLCHECK(nccl_uct_iface_set_rtr_mode(w->uct_iface, NULL));
  NCCLCHECK(nccl_uct_iface_set_handler(w->uct_iface, NCCL_UCT_AM_RTS,
                                       nccl_uct_rts_callback));
  NCCLCHECK(nccl_uct_iface_set_handler(w->uct_iface, NCCL_UCT_AM_ATS,
                                       nccl_uct_ats_callback));
  return ncclSuccess;

fail:
  return ncclSystemError;
}

static nccl_uct_worker_t *nccl_uct_worker_get(nccl_uct_context_t *context,
                                              int dev) {
  nccl_uct_worker_t *w;

  pthread_mutex_lock(&nccl_uct_lock);

  for (w = context->worker_list; w != NULL; w = w->next) {
    if (w->id.dev == dev) {
      goto found;
    }
  }

  w = calloc(1, sizeof(*w));
  if (w == NULL) {
    WARN("Failed worker allocation: dev=%d", dev);
    goto out;
  }

  if (nccl_uct_worker_create(w, context, dev) != ncclSuccess) {
    free(w);
    w = NULL;
    goto out;
  }

  w->next              = context->worker_list;
  context->worker_list = w;

found:
  w->count++;
out:
  pthread_mutex_unlock(&nccl_uct_lock);
  return w;
}

static ncclResult_t nccl_uct_listen(int dev, void *listen_handle,
                                    void **listen_comm) {
  nccl_uct_listen_handle_t *handle = listen_handle;
  nccl_uct_listen_comm_t *l_comm   = calloc(1, sizeof(*l_comm));
  nccl_uct_comm_t *accept_comm;
  union ncclSocketAddress addr;

  if (l_comm == NULL) {
    WARN("Failed to alloc UCT listener(dev=%d)", dev);
    return ncclSystemError;
  }

  NCCL_STATIC_ASSERT(sizeof(nccl_uct_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                     "UCT listen handle is too big");

  NCCLCHECK(ncclSocketInit(&l_comm->sock, &context.if_addr,
                           NCCL_UCT_LISTEN_HANDLE_MAGIC, ncclSocketTypeNetIb,
                           NULL, 1));
  NCCLCHECK(ncclSocketListen(&l_comm->sock));
  NCCLCHECK(ncclSocketGetAddr(&l_comm->sock, &addr));

  l_comm->uct_worker = nccl_uct_worker_get(&context, dev);
  if (l_comm->uct_worker == NULL) {
    WARN("Failed to create worker for listener dev=%d", dev);
    return ncclSystemError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&accept_comm, sizeof(*accept_comm)));

  l_comm->comm    = accept_comm;
  l_comm->context = &context;
  l_comm->dev     = dev;
  l_comm->id      = context.listener_count++;

  *listen_comm = l_comm;

  memset(handle, 0, sizeof(*handle));
  handle->magic         = NCCL_UCT_LISTEN_HANDLE_MAGIC;
  handle->listener.id   = l_comm->id;
  handle->listener.addr = addr;
  handle->comm          = accept_comm;

  WARN("Listening id=%d dev=%d comm=%p", l_comm->id, dev, handle->comm);
  return ncclSuccess;
}

static ncclResult_t nccl_uct_close_listen(void *listen_comm) {
  nccl_uct_listen_comm_t *comm = listen_comm;

  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

static ncclResult_t nccl_uct_comm_init(nccl_uct_comm_t *comm,
                                       nccl_uct_context_t *context,
                                       nccl_uct_worker_t *worker, int dev,
                                       const nccl_uct_comm_t *remote_comm) {
  int i;

  if (worker == NULL) {
    worker = nccl_uct_worker_get(context, dev);
  }

  for (i = 0; i < NCCL_UCT_RING_HASH_SIZE; i++) {
    nccl_uct_ring_init(&comm->exp[i]);
    nccl_uct_ring_init(&comm->unexp[i]);
  }

  comm->get.first = 0;
  comm->get.last  = 0;
  comm->free_req  = NULL;
  comm->req_count = 0;

  comm->uct_worker = worker;
  if (comm->uct_worker == NULL) {
    return ncclSystemError;
  }

  comm->dev         = dev;
  comm->context     = context;
  comm->remote_comm = remote_comm;
  comm->uct_iface   = comm->uct_worker->uct_iface;
  comm->uct_ep      = nccl_uct_ep_create(comm->uct_iface);
  if (comm->uct_ep == NULL) {
    return ncclSystemError;
  }
  comm->uct_ep_am   = nccl_uct_ep_create(comm->uct_iface);
  if (comm->uct_ep_am == NULL) {
    return ncclSystemError;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_comm_gpu_flush_init(nccl_uct_comm_t *comm) {
  ucs_status_t status;

  comm->gpu_flush.enabled = (nccl_p2p_gdr_support(comm->dev) == ncclSuccess) ||
                            (nccl_p2p_dmabuf_support(comm->dev) == ncclSuccess);

  if (!comm->gpu_flush.enabled) {
    return ncclSuccess;
  }

  comm->gpu_flush.uct_ep = nccl_uct_ep_create(comm->uct_iface);
  if (comm->gpu_flush.uct_ep == NULL) {
    return ncclSystemError;
  }

  NCCLCHECK(nccl_uct_ep_addr_set(&comm->gpu_flush.addr, comm,
                                 comm->gpu_flush.uct_ep));
  NCCLCHECK(
      nccl_uct_ep_connect_to_ep(comm->gpu_flush.uct_ep, &comm->gpu_flush.addr));

  /* Remove when replaced by bcopy */
  status = uct_md_mem_reg(comm->uct_iface->md, (void*)comm->gpu_flush.mem,
                          sizeof(comm->gpu_flush.mem), UCT_MD_MEM_ACCESS_ALL,
                          &comm->gpu_flush.memh);
  return status == UCS_OK ? ncclSuccess : ncclSystemError;
}

static ncclResult_t nccl_uct_connect(int dev, void *listen_handle,
                                     void **send_comm,
                                     ncclNetDeviceHandle_t **sendDevComm) {
  int ready                        = 0;
  nccl_uct_listen_handle_t *handle = listen_handle;
  nccl_uct_stage_t *stage          = &handle->stage;
  nccl_uct_comm_t *comm            = stage->comm;
  nccl_uct_comm_addr_t addr;

  *send_comm = NULL;

  switch (stage->state) {
  case NCCL_UCT_START:
    NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(*comm)));
    NCCLCHECK(nccl_uct_comm_init(comm, &context, NULL, dev, handle->comm));
    NCCLCHECK(ncclSocketInit(&comm->sock, &handle->listener.addr, handle->magic,
                             ncclSocketTypeNetIb, NULL, 1));
    NCCLCHECK(ncclSocketConnect(&comm->sock));

    stage->comm  = comm;
    stage->state = NCCL_UCT_CONNECT;
    /* fallthrough */

  case NCCL_UCT_CONNECT:
    NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
    if (!ready) {
      return ncclSuccess;
    }

    NCCLCHECK(nccl_uct_ep_addr_set(&addr.rma, comm, comm->uct_ep));
    NCCLCHECK(nccl_uct_ep_addr_set(&addr.am, comm, comm->uct_ep_am));
    NCCLCHECK(ncclSocketSend(&comm->sock, &addr, sizeof(addr)));
    NCCLCHECK(ncclSocketSend(&comm->sock, &comm, sizeof(comm)));

    stage->offset = 0;
    stage->state  = NCCL_UCT_RECEIVE_ADDR;
    /* fallthrough */

  case NCCL_UCT_RECEIVE_ADDR:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &comm->addr,
                                 sizeof(comm->addr), &stage->offset));
    if (stage->offset != sizeof(comm->addr)) {
      return ncclSuccess; /* In progress */
    }

    ready = 1;
    NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep, &comm->addr.rma));
    NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep_am, &comm->addr.am));
    NCCLCHECK(ncclSocketSend(&comm->sock, &ready, sizeof(ready)));

    *send_comm   = comm;
    stage->state = NCCL_UCT_DONE;
    WARN("Connected comm=%p remote_comm=%p listener_id=%d", comm,
         comm->remote_comm, handle->listener.id);
    break;

  default:
    WARN("UCT connnect for dev=%d using unsupported state %d", dev,
         stage->state);
    return ncclSystemError;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_accept(void *listen_comm, void **recv_comm,
                                    ncclNetDeviceHandle_v7_t **recvDevComm) {
  nccl_uct_listen_comm_t *l_comm = listen_comm;
  nccl_uct_stage_t *stage        = &l_comm->stage;
  nccl_uct_comm_t *comm          = stage->comm;
  nccl_uct_comm_addr_t addr;
  int ready;

  *recv_comm = NULL;

  switch (stage->state) {
  case NCCL_UCT_START:
    comm = l_comm->comm;

    NCCLCHECK(ncclSocketInit(&comm->sock, NULL, NCCL_SOCKET_MAGIC,
                             ncclSocketTypeUnknown, NULL, 0));
    NCCLCHECK(ncclSocketAccept(&comm->sock, &l_comm->sock));
    NCCLCHECK(nccl_uct_comm_init(comm, l_comm->context, l_comm->uct_worker,
                                 l_comm->dev, NULL));
    NCCLCHECK(nccl_uct_comm_gpu_flush_init(comm));

    stage->comm  = comm;
    stage->state = NCCL_UCT_ACCEPT;
    /* fallthrough */

  case NCCL_UCT_ACCEPT:
    NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
    if (!ready) {
      return ncclSuccess;
    }

    NCCLCHECK(nccl_uct_ep_addr_set(&addr.rma, comm, comm->uct_ep));
    NCCLCHECK(nccl_uct_ep_addr_set(&addr.am, comm, comm->uct_ep_am));
    NCCLCHECK(ncclSocketSend(&comm->sock, &addr, sizeof(addr)));

    stage->offset = 0;
    stage->state  = NCCL_UCT_RECEIVE_ADDR;
    /* fallthrough */

  case NCCL_UCT_RECEIVE_ADDR:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &comm->addr,
                                 sizeof(comm->addr), &stage->offset));
    if (stage->offset != sizeof(comm->addr)) {
      return ncclSuccess;
    }

    NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep, &comm->addr.rma));
    NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep_am, &comm->addr.am));

    stage->offset = 0;
    stage->state  = NCCL_UCT_RECEIVE_COMM;
    /* fallthrough */

  case NCCL_UCT_RECEIVE_COMM:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock,
                                 &comm->remote_comm, sizeof(comm->remote_comm),
                                 &stage->offset));
    if (stage->offset != sizeof(comm->remote_comm)) {
      return ncclSuccess;
    }

    stage->offset = 0;
    stage->ready  = 0;
    stage->state  = NCCL_UCT_RX_READY;
    /* fallthrough */

  case NCCL_UCT_RX_READY:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &stage->ready,
                                 sizeof(stage->ready), &stage->offset));
    if (stage->offset != sizeof(ready)) {
      return ncclSuccess;
    }
    if (stage->ready != 1) {
      WARN("Accepted comm=%p invalid status (ready=%d)", stage->ready);
      return ncclSystemError;
    }

    *recv_comm   = comm;
    stage->state = NCCL_UCT_DONE;
    WARN("Accepted comm=%p remote_comm=%p listener_id=%d", comm,
         comm->remote_comm, l_comm->id);
    break;

  default:
    WARN("UCT accept for dev=%d using unsupported state %d", l_comm->dev,
         stage->state);
    return ncclSystemError;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_reg_mr(void *reg_comm, void *data, size_t size,
                                    int type, void **mhandle) {
  nccl_uct_comm_t *comm = reg_comm;
  uct_component_h comp  = comm->uct_iface->comp;
  uct_md_h md           = comm->uct_iface->md;
  intptr_t addr         = (intptr_t)data;
  size_t rkey_size      = comm->uct_iface->rkey_packed_size;
  nccl_uct_memh_t *uct_memh;
  ucs_status_t status;

  NCCLCHECK(ncclIbMalloc((void**)&uct_memh, sizeof(*uct_memh) + rkey_size));
  uct_memh->comm = comm;

  /* Use integral pages */
  size += addr & (NCCL_UCT_REG_ALIGN - 1);
  size  = (size + NCCL_UCT_REG_ALIGN - 1) & ~(NCCL_UCT_REG_ALIGN - 1);
  addr &= ~(NCCL_UCT_REG_ALIGN - 1);

  /* Register memory */
  status = uct_md_mem_reg(md, (void*)addr, size, UCT_MD_MEM_ACCESS_ALL,
                          &uct_memh->memh);
  if (status != UCS_OK) {
    WARN("Failed to register %p/%zu on comm %p", addr, size, comm);
    return ncclSystemError;
  }

  /* Pack memory */
  status = uct_md_mkey_pack(md, uct_memh->memh, uct_memh->rkey);
  if (status != UCS_OK) {
    WARN("Failed to pack rkey for %p/%zu on comm %p", addr, size, comm);
    return ncclSystemError;
  }

  /* Unpack rkey from sender side */
  status = uct_rkey_unpack(comp, uct_memh->rkey, &uct_memh->bundle);
  if (status != UCS_OK) {
    WARN("Failed to unpack rkey: error %d", status);
    return ncclInternalError;
  }

  *mhandle = uct_memh;
  return ncclSuccess;
}

static ncclResult_t nccl_uct_reg_mr_dmabuf(void *reg_comm, void *data,
                                           size_t size, int type,
                                           uint64_t offset, int fd,
                                           void **mhandle) {
  return nccl_uct_reg_mr(reg_comm, data, size, type, mhandle);
}

static ncclResult_t nccl_uct_dereg_mr(void *dereg_comm, void *mhandle) {
  nccl_uct_comm_t *comm     = dereg_comm;
  uct_component_h comp      = comm->uct_iface->comp;
  nccl_uct_memh_t *uct_memh = mhandle;
  ucs_status_t status;

  assert(uct_memh->memh != UCT_MEM_HANDLE_NULL);
  assert(uct_memh->comm == comm);

  status = uct_rkey_release(comp, &uct_memh->bundle);
  if (status != UCS_OK) {
    WARN("Failed to release rkey bundle: error %d", status);
  }

  status = uct_md_mem_dereg(comm->uct_iface->md, uct_memh->memh);
  if (status != UCS_OK) {
    WARN("Failed to deregister memh %p on comm %p: error %d", uct_memh, comm,
         status);
    return ncclSystemError;
  }

  uct_memh->comm = NULL;
  free(uct_memh);
  return ncclSuccess;
}

/* Outcome is either send_atp equal to 1 or 0 */
static void nccl_uct_send_atp(nccl_uct_comm_t *comm, nccl_uct_rdesc_t *rdesc) {
  ucs_status_t status;
  nccl_uct_atp_t atp;
  int i;

  assert(rdesc->send_atp == 1);

  status = uct_ep_fence(comm->uct_ep->ep, 0);
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

  status = uct_ep_am_short(comm->uct_ep->ep, NCCL_UCT_AM_ATP,
                           (uint64_t)comm->remote_comm, &atp, sizeof(atp));
  if (status == UCS_OK) {
    rdesc->send_atp = 0;
  }
}

static ncclResult_t nccl_uct_send(nccl_uct_comm_t *comm, void *data, int size,
                                  nccl_uct_memh_t *uct_memh,
                                  nccl_uct_rdesc_t *rdesc, int i,
                                  void **request) {
  ucs_status_t status;
  uct_iov_t iov;
  nccl_uct_req_t *req;

  *request = NULL;

  /* Details for local data */
  iov.buffer = data;
  iov.length = size;
  iov.memh   = uct_memh->memh;
  iov.stride = iov.length;
  iov.count  = 1;

  assert(size <= rdesc->desc.chunk[i].size);

  req = nccl_uct_rdesc_get_req(rdesc, i, size); /* NCCL request */

  status = uct_ep_put_zcopy(comm->uct_ep->ep, &iov, 1,
                            (uint64_t)rdesc->desc.chunk[i].data,
                            rdesc->desc.chunk[i].rkey, &req->completion);

  if (status == UCS_OK) {
    req->completion.count--;
  } else if (status != UCS_INPROGRESS) {
    return ncclSuccess;
  }

  rdesc->desc.chunk[i].matched = 1;
  --rdesc->send_atp;

  if (rdesc->send_atp == 1) {
    nccl_uct_comm_rdesc_del(rdesc); /* all ->isend() were now matched */
    nccl_uct_send_atp(comm, rdesc);
  }

  *request = req;
  return ncclSuccess;
}

static ncclResult_t nccl_uct_isend(void *send_comm, void *data, int size,
                                   int tag, void *mhandle, void **request) {
  nccl_uct_comm_t *comm = send_comm;
  nccl_uct_rdesc_t *rdesc;
  int i;

  *request = NULL;

  for (rdesc = comm->rdesc_list.head; rdesc != NULL; rdesc = rdesc->next) {
    for (i = 0; i < rdesc->desc.count; i++) {
      if (rdesc->desc.chunk[i].matched || (rdesc->desc.chunk[i].tag != tag)) {
        continue;
      }

      return nccl_uct_send(comm, data, size, mhandle, rdesc, i, request);
    }
  }

  /* Progress here to make sure we receive non-solicited RTRs */
  uct_worker_progress(comm->uct_worker->worker);
  return ncclSuccess;
}

static ncclResult_t nccl_uct_irecv(void *recv_comm, int n, void **data,
                                   int *sizes, int *tags, void **mhandles,
                                   void **request) {
  nccl_uct_comm_t *comm      = recv_comm;
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandles;
  nccl_uct_rdesc_t *rdesc;
  ucs_status_t status;

  assert(n <= NCCL_UCX_UCT_MAX_RECVS);

  rdesc = nccl_uct_comm_rdesc_get(comm);
  if (rdesc == NULL) {
    return ncclInternalError;
  }

  nccl_uct_rdesc_set(rdesc, comm->rdesc_id++, n, data, sizes, tags, uct_memh);

  status = uct_ep_am_short(comm->uct_ep->ep, NCCL_UCT_AM_RTR,
                           (uint64_t)comm->remote_comm, &rdesc->desc,
                           nccl_uct_rdesc_size(n));
  if (status != UCS_OK) {
    nccl_uct_comm_rdesc_put(rdesc);
    *request = NULL;
  } else {
    /* Wait for receiving ATP */
    *request = nccl_uct_rdesc_get_req(rdesc, 0, NCCL_UCT_REQ_IRECV);
  }

  return ncclSuccess;
}

/* TODO reactivate / port iflush */
static ncclResult_t nccl_uct_iflush(void *recv_comm, int n, void **data,
                                    int *sizes, void **mhandle,
                                    void **request) {
  int last                   = -1;
  nccl_uct_comm_t *comm      = recv_comm;
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandle;
  nccl_uct_rdesc_t *rdesc;
  nccl_uct_req_t *req;
  ucs_status_t status;
  uct_iov_t iov;
  int i;

  if (comm->gpu_flush.enabled) {
    for (i = 0; i < n; i++) {
      if (sizes[i]) {
        last = i;
      }
    }
  }

  return ncclSuccess;
  if (last == -1) {
    return ncclSuccess;
  }

  rdesc = nccl_uct_comm_rdesc_get(comm);
  if (rdesc == NULL) {
    return ncclInternalError;
  }

  nccl_uct_rdesc_set(rdesc, ~0, 0, NULL, NULL, NULL, NULL);
  /* Wait for local GET completion */
  req = nccl_uct_rdesc_get_req(rdesc, 0, NCCL_UCT_REQ_IFLUSH);

  iov.buffer = comm->gpu_flush.mem;
  iov.length = sizeof(comm->gpu_flush.mem);
  iov.memh   = comm->gpu_flush.memh;
  iov.stride = 0;
  iov.count  = 1;

  status = uct_ep_get_zcopy(comm->gpu_flush.uct_ep->ep, &iov, 1,
                            (uint64_t)data[last], uct_memh[last]->bundle.rkey,
                            &req->completion);
  if (status == UCS_OK) {
    *request = NULL;
    nccl_uct_comm_rdesc_put(rdesc);
  } else if (status == UCS_INPROGRESS) {
    *request = req;
  } else {
    WARN("Failed to flush local ep comm=%p status=%d", comm, status);
    return ncclInternalError;
  }

  return ncclSuccess;
}

/* TODO reactivate / port iflush */
static ncclResult_t nccl_uct_rd_iflush(void *recv_comm, int n, void **data,
                                       int *sizes, void **mhandle,
                                       void **request) {
  int last                   = -1;
  nccl_uct_comm_t *comm      = recv_comm;
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandle;
  nccl_uct_rd_req_t *req;
  ucs_status_t status;
  uct_iov_t iov;
  int i;

  if (comm->gpu_flush.enabled) {
    for (i = 0; i < n; i++) {
      if (sizes[i]) {
        last = i;
      }
    }
  }

  if (last == -1) {
    return ncclSuccess;
  }

  req = nccl_uct_rd_req_alloc(comm);
  if (req == NULL) {
    *request = NULL;
    return ncclSuccess;
  }

  req->send_rts          = -1;
  req->completion.func   = nccl_uct_empty_callback;
  req->completion.count  = 1;
  req->completion.status = UCS_OK;

  iov.buffer = comm->gpu_flush.mem;
  iov.length = sizeof(comm->gpu_flush.mem);
  iov.memh   = comm->gpu_flush.memh;
  iov.stride = 0;
  iov.count  = 1;

  status = uct_ep_get_zcopy(comm->gpu_flush.uct_ep->ep, &iov, 1,
                            (uint64_t)data[last], uct_memh[last]->bundle.rkey,
                            &req->completion);
  if (status == UCS_OK) {
    nccl_uct_rd_req_free(req);
    *request = NULL;
  } else if (status == UCS_INPROGRESS) {
    *request = req;
  } else {
    WARN("Failed to flush local ep comm=%p status=%d", comm, status);
    return ncclInternalError;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_test(void *request, int *done, int *sizes) {
  nccl_uct_req_t *req     = request;
  nccl_uct_rdesc_t *rdesc = req->rdesc;
  nccl_uct_comm_t *comm   = rdesc->comm;

  uct_worker_progress(comm->uct_worker->worker);

  *done = 0;

  if (rdesc->send_atp == 1) {
    /* Slowpath */
    nccl_uct_send_atp(comm, rdesc);

    if (rdesc->send_atp && rdesc->nccl_usage == 1) {
      /* Keep the last isend request until ATP is out */
      return ncclSuccess;
    }
  }

  if (req->completion.count > 0) {
    return ncclSuccess;
  }

  *done = 1;

  if (req->size == NCCL_UCT_REQ_IRECV) {
    assert(&rdesc->reqs[0] == req);
    if (sizes != NULL) {
      memcpy(sizes, rdesc->sizes, rdesc->desc.count * sizeof(*sizes));
    }
  } else if (req->size == NCCL_UCT_REQ_IFLUSH) {
    assert(&rdesc->reqs[0] == req);
  } else {
    /* ->isend() request */
    assert(req->size > -1);
    if (sizes != NULL) {
      sizes[0] = req->size;
    }
  }

  if (--rdesc->nccl_usage < 1) {
    assert(rdesc->send_atp == 0);
    assert(rdesc->nccl_usage == 0);
    nccl_uct_comm_rdesc_put(rdesc);
  }

  return ncclSuccess;
}

static void nccl_uct_rd_send(nccl_uct_rd_req_t *req)
{
  ucs_status_t status;

  assert(req->send_rts == 1);
  assert(req->completion.count == 2);

  status = uct_ep_am_short(req->comm->uct_ep->ep, NCCL_UCT_AM_RTS,
                           (uint64_t)req->comm->remote_comm, &req->rts,
                           sizeof(req->rts));
  if (status == UCS_OK) {
    req->completion.count--;
  }
}

/* TODO: batch rts in one send/write in test */
ncclResult_t nccl_uct_rd_isend(void *send_comm, void *data, int size,
                                      int tag, void *mhandle, void **request) {

  nccl_uct_comm_t *comm     = send_comm;
  nccl_uct_memh_t *uct_memh = mhandle;
  nccl_uct_rd_req_t *req;

  req = nccl_uct_rd_req_alloc(comm);
  if (req != NULL) {
    req->send_rts         = 1;
    req->completion.count = 2;
    req->rts.tag          = tag;
    req->rts.size         = size;
    req->rts.data         = data;
    req->rts.u.rkey       = uct_memh->bundle.rkey;
    req->rts.opaque       = req;
    req->count            = 1;
		/* TODO: add sequence number and check it */

    nccl_uct_rd_send(req);
  }

	if (req->completion.count == 2) {
    nccl_uct_rd_req_free(req);
		*request = NULL;
	} else {
		*request = req;
  }
  return ncclSuccess;
}

ncclResult_t nccl_uct_rd_test(void *request, int *done, int *sizes) {
  nccl_uct_rd_req_t *req  = request;
  nccl_uct_comm_t *comm   = req->comm;
  int i;

  while (uct_worker_progress(comm->uct_worker->worker)) {}

  nccl_uct_match_drain(comm);

  if (req->completion.count > 0) {
    if (req->send_rts && req->completion.count == 2) {
      nccl_uct_rd_send(req);
    }

    if ((req->send_rts == 0) && (req->completion.count == 1)) {
      nccl_uct_rd_send_ats(req);
    }

    if (req->completion.count > 0) {
      *done = 0;
      return ncclSuccess;
    }
  }

  *done = 1;

  if ((sizes != NULL) && (req->send_rts > -1)) {
    if (req->send_rts) {
      /* TODO remove rts member and use recv[0] aliased */
      sizes[0] = req->rts.size;
    } else {
      for (i = 0; i < req->count; i++) {
        sizes[i] = req->recv[i].size;
      }
    }
  }

  nccl_uct_rd_req_free(req);

  return ncclSuccess;
}

static void nccl_uct_worker_destroy(nccl_uct_worker_t *w) {
  nccl_uct_iface_close(w->uct_iface);
  uct_worker_destroy(w->worker);
  ucs_async_context_destroy(w->async);
  free(w);
}

static void nccl_uct_worker_put(nccl_uct_worker_t *worker) {
  nccl_uct_worker_t **wp;

  pthread_mutex_lock(&nccl_uct_lock);
  if (--worker->count < 1) {
    assert(worker->count == 0);
    for (wp = &worker->context->worker_list; *wp != NULL; wp = &(*wp)->next) {
      if (*wp == worker) {
        *wp = worker->next;
        nccl_uct_worker_destroy(worker);
        break;
      }
    }
  }
  pthread_mutex_unlock(&nccl_uct_lock);
}

static ncclResult_t nccl_uct_close(void *close_comm) {
  nccl_uct_comm_t *comm = close_comm;
  nccl_uct_rdesc_t *rdesc;
  nccl_uct_rd_req_t *req;

  nccl_uct_ep_destroy(comm->uct_ep);
  nccl_uct_ep_destroy(comm->uct_ep_am);
  if (comm->gpu_flush.uct_ep != NULL) {
    nccl_uct_ep_destroy(comm->gpu_flush.uct_ep);
    (void)uct_md_mem_dereg(comm->uct_iface->md, comm->gpu_flush.memh);
  }
  nccl_uct_worker_put(comm->uct_worker);

  while ((rdesc = comm->free_rdesc) != NULL) {
    comm->free_rdesc = rdesc->next;
    free(rdesc);
  }

  while ((req = comm->free_req) != NULL) {
    comm->free_req = req->next;
    free(req);
  }

  assert(comm->rdesc_list.head == NULL);
  assert(comm->rdesc_list.tail == NULL);
  assert(comm->rdesc_alloc == 0);
  assert(comm->get.first == comm->get.last);
  assert(comm->req_count == 0);
  free(comm);

  return ncclSuccess;
}

ncclResult_t nccl_uct_get_properties_v7(int dev,
                                        ncclNetProperties_v7_t *props_v7) {
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_uct_get_properties(dev, &props);
  if (ret != ncclSuccess) {
    return ret;
  }

  props_v7->name             = props.name;
  props_v7->pciPath          = props.pciPath;
  props_v7->guid             = props.guid;
  props_v7->ptrSupport       = props.ptrSupport;
  props_v7->speed            = props.speed;
  props_v7->latency          = props.latency;
  props_v7->port             = props.port;
  props_v7->maxComms         = props.maxComms;
  props_v7->maxRecvs         = props.maxRecvs;
  props_v7->netDeviceType    = props.netDeviceType;
  props_v7->netDeviceVersion = props.netDeviceVersion;

  return ncclSuccess;
}

static ncclResult_t nccl_uct_reg_mr_v7(void *comm, void *data, int size,
                                       int type, void **mhandle) {
  return nccl_uct_reg_mr(comm, data, (size_t)size, type, mhandle);
}

static ncclResult_t
nccl_uct_get_properties_v6(int dev, ncclNetProperties_v6_t *props_v6) {
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_uct_get_properties(dev, &props);
  if (ret != ncclSuccess) {
    return ret;
  }

  props_v6->name       = props.name;
  props_v6->pciPath    = props.pciPath;
  props_v6->guid       = props.guid;
  props_v6->ptrSupport = props.ptrSupport;
  props_v6->speed      = props.speed;
  props_v6->latency    = props.latency;
  props_v6->port       = props.port;
  props_v6->maxComms   = props.maxComms;
  props_v6->maxRecvs   = props.maxRecvs;

  return ncclSuccess;
}

static ncclResult_t nccl_uct_connect_v6(int dev, void *handle,
                                        void **send_comm) {
  ncclNetDeviceHandle_v7_t *dev_handle = NULL;
  return nccl_uct_connect(dev, handle, send_comm, &dev_handle);
}

static ncclResult_t nccl_uct_accept_v6(void *listen_comm, void **recv_comm) {
  ncclNetDeviceHandle_v7_t *dev_handle = NULL;
  return nccl_uct_accept(listen_comm, recv_comm, &dev_handle);
}

ncclNet_v8_t ucxUctPlugin_v8 = {
  .name          = "UCX-UCT",
  .init          = nccl_uct_init,
  .devices       = nccl_uct_devices,
  .getProperties = nccl_uct_get_properties,
  .listen        = nccl_uct_listen,
  .connect       = nccl_uct_connect,
  .accept        = nccl_uct_accept,
  .regMr         = nccl_uct_reg_mr,
  .regMrDmaBuf   = nccl_uct_reg_mr_dmabuf,
  .deregMr       = nccl_uct_dereg_mr,
  .isend         = nccl_uct_rd_isend,
  .irecv         = nccl_uct_rd_irecv,
  .iflush        = nccl_uct_rd_iflush,
  .test          = nccl_uct_rd_test,
  .closeSend     = nccl_uct_close,
  .closeRecv     = nccl_uct_close,
  .closeListen   = nccl_uct_close_listen,
  .getDeviceMr   = NULL,
  .irecvConsumed = NULL
};

ncclNet_v7_t ucxUctPlugin_v7 = {
  .name          = "UCX-UCT",
  .init          = nccl_uct_init,
  .devices       = nccl_uct_devices,
  .getProperties = nccl_uct_get_properties_v7,
  .listen        = nccl_uct_listen,
  .connect       = nccl_uct_connect,
  .accept        = nccl_uct_accept,
  .regMr         = nccl_uct_reg_mr_v7,
  .regMrDmaBuf   = nccl_uct_reg_mr_dmabuf,
  .deregMr       = nccl_uct_dereg_mr,
  .isend         = nccl_uct_isend,
  .irecv         = nccl_uct_irecv,
  .iflush        = nccl_uct_iflush,
  .test          = nccl_uct_test,
  .closeSend     = nccl_uct_close,
  .closeRecv     = nccl_uct_close,
  .closeListen   = nccl_uct_close_listen,
  .getDeviceMr   = NULL,
  .irecvConsumed = NULL
};

ncclNet_v6_t ucxUctPlugin_v6 = {
  .name          = "UCX-UCT",
  .init          = nccl_uct_init,
  .devices       = nccl_uct_devices,
  .getProperties = nccl_uct_get_properties_v6,
  .listen        = nccl_uct_listen,
  .connect       = nccl_uct_connect_v6,
  .accept        = nccl_uct_accept_v6,
  .regMr         = nccl_uct_reg_mr_v7,
  .regMrDmaBuf   = nccl_uct_reg_mr_dmabuf,
  .deregMr       = nccl_uct_dereg_mr,
  .isend         = nccl_uct_isend,
  .irecv         = nccl_uct_irecv,
  .iflush        = nccl_uct_iflush,
  .test          = nccl_uct_test,
  .closeSend     = nccl_uct_close,
  .closeRecv     = nccl_uct_close,
  .closeListen   = nccl_uct_close_listen
};

ncclNet_v5_t ucxUctPlugin_v5 = {
  .name          = "UCX-UCT",
  .init          = nccl_uct_init,
  .devices       = nccl_uct_devices,
  .getProperties = nccl_uct_get_properties_v6,
  .listen        = nccl_uct_listen,
  .connect       = nccl_uct_connect_v6,
  .accept        = nccl_uct_accept_v6,
  .regMr         = nccl_uct_reg_mr_v7,
  .deregMr       = nccl_uct_dereg_mr,
  .isend         = nccl_uct_isend,
  .irecv         = nccl_uct_irecv,
  .iflush        = nccl_uct_iflush,
  .test          = nccl_uct_test,
  .closeSend     = nccl_uct_close,
  .closeRecv     = nccl_uct_close,
  .closeListen   = nccl_uct_close_listen
};
