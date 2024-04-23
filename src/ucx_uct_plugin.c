/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdint.h>
#include <unistd.h>

#include "socket.h"
#include "p2p_plugin.h"

#include <uct/api/uct.h>


#define NCCL_UCX_UCT_MAX_RECVS       NCCL_NET_IB_MAX_RECVS
#define NCCL_UCT_LISTEN_HANDLE_MAGIC 0x43cf19ed91abdb85
#define NCCL_UCT_REG_ALIGN           4096

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
    NCCL_UCT_AM_ATP = 15
} nccl_uct_am_type_t;

typedef enum {
    NCCL_UCT_REQ_IRECV  = -1,
    NCCL_UCT_REQ_IFLUSH = -2
} nccl_uct_request_type_t;

/* UCT EP address to exchange and connect to */
typedef struct {
    uint8_t                 dev_addr_size;
    uint8_t                 ep_addr_size;
    uint8_t                 data[64];
} nccl_uct_ep_addr_t;

typedef struct {
    uct_iface_h             iface;
    uct_md_h                md;
    uct_component_h         comp;
    void                    *addr;
    size_t                  addr_size;
    void                    *dev_addr;
    size_t                  dev_addr_size;
    size_t                  ep_addr_size;
    size_t                  rkey_packed_size;

    size_t                  am_max_short;
} nccl_uct_iface_t;

struct nccl_uct_context;

typedef struct nccl_uct_worker {
    struct nccl_uct_worker *next;
    struct {
        pthread_t           thread;
        int                 dev;
    } id;

    int                     count;
    ucs_async_context_t     *async;
    uct_worker_h            worker;
    nccl_uct_iface_t        *uct_iface;
    struct nccl_uct_context *context;
} nccl_uct_worker_t;

typedef struct {
    uct_ep_h                ep;
    uct_ep_addr_t           *addr;
    size_t                  addr_size;
    nccl_uct_iface_t        *uct_iface;
    uint8_t                 data[];
} nccl_uct_ep_t;

struct nccl_uct_rdesc;
struct nccl_uct_comm;

/* On-the-wire descriptor of a posted receive request entry */
typedef struct {
    int                     tag;
    int                     size;
    void                    *data;
    int                     matched;
    uct_rkey_t              rkey;
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

/*
 * NCCL local request handler to progress:
 * - size -1 for multi receive
 * - size -2 for flush
 * - size > 0 for send
 */
typedef struct {
    int                   size;
    struct nccl_uct_rdesc *rdesc;
} nccl_uct_req_t;

/* Pending receive descriptor either on the receive or sending side */
typedef struct nccl_uct_rdesc {
    uct_completion_t      completion; /* pending rx_ATP or all PUTs+tx_ATP */
    int                   nccl_usage; /* NCCL requests not finished/started */
    int                   send_atp;   /* >1 pending isend, ==1 pending atp send */

    struct nccl_uct_rdesc *prev;      /* comm's linked list */
    struct nccl_uct_rdesc *next;

    struct nccl_uct_comm  *comm;
    nccl_uct_rdesc_hdr_t  desc;
    nccl_uct_chunk_t      storage[NCCL_UCX_UCT_MAX_RECVS]; /* Don't use directly */
    nccl_uct_req_t        reqs[NCCL_UCX_UCT_MAX_RECVS];    /* NCCL requests */
    int                   sizes[NCCL_UCX_UCT_MAX_RECVS];   /* ATP received sizes */
} nccl_uct_rdesc_t;

/* All the remote addresses for the communicator */
typedef struct nccl_uct_comm_addr {
    nccl_uct_ep_addr_t      rma;
    /* TODO: Add multi-QP here */
} nccl_uct_comm_addr_t;

/* Either Receiver or Sender communicator, connected to one peer */
typedef struct nccl_uct_comm {
    struct ncclSocket       sock;
    struct nccl_uct_context *context;
    int                     dev;

    nccl_uct_worker_t       *uct_worker;
    nccl_uct_iface_t        *uct_iface;
    nccl_uct_ep_t           *uct_ep;

    nccl_uct_comm_addr_t    addr;         /* Remote addresses */

    nccl_uct_rdesc_t        *free_rdesc;  /* Available rdesc for reuse */
    uint64_t                rdesc_id;     /* Next sequence number to use */

    const struct nccl_uct_comm *remote_comm; /* Cookie received while connecting */

    /* Local GET on current device */
    struct {
        int                 enabled;
        nccl_uct_ep_t       *uct_ep;    /* Locally read from HCA */
        nccl_uct_ep_addr_t  addr;

        uint8_t             mem[65];    /* Dummy memory to read into */
        uct_mem_h           memh;
    } gpu_flush;

    /* Received RTRs: used by Sender communicator in ->isend() */
    struct {
        nccl_uct_rdesc_t    *head;
        nccl_uct_rdesc_t    *tail;
    } rdesc_list;
} nccl_uct_comm_t;

/* State tracking used while connecting/accepting only */
typedef struct {
    nccl_uct_state_t        state;
    nccl_uct_comm_t         *comm;  /* current communicator being created */
    int                     offset; /* for Socket reading */
    int                     ready;  /* accept must complete after connect */
} nccl_uct_stage_t;

/* Memory registration handle in NCCL UCT plugin returned by ->regMR() */
typedef struct {
    uct_mem_h               memh;
    nccl_uct_comm_t         *comm;
    uct_rkey_bundle_t       bundle;
    uint8_t                 rkey[];
} nccl_uct_memh_t;

/* On-the-wire handle passed OOB by NCCL from listener to connector */
typedef struct {
    uint64_t                    magic;
    struct {
        union ncclSocketAddress addr;
        uint32_t                id;
    } listener;
    nccl_uct_comm_t             *comm; /* Created communicator in accept */
    nccl_uct_stage_t            stage; /* Used by connector */
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


static pthread_mutex_t nccl_uct_lock = PTHREAD_MUTEX_INITIALIZER;

static nccl_uct_context_t context = {
    .tl_name   = "rc_mlx5",
    .dev_count = -1
};

static const uct_device_addr_t *
nccl_uct_ep_addr_dev(const nccl_uct_ep_addr_t *addr)
{
    return (uct_device_addr_t *)addr->data;
}

static const uct_ep_addr_t *nccl_uct_ep_addr_ep(const nccl_uct_ep_addr_t *addr)
{
    return (uct_ep_addr_t *)(addr->data + addr->dev_addr_size);
}

static ncclResult_t nccl_uct_ep_addr_set(nccl_uct_ep_addr_t *addr,
                                         const nccl_uct_comm_t *comm,
                                         const nccl_uct_ep_t *uct_ep)
{
    nccl_uct_iface_t *uct_iface = comm->uct_iface;
    size_t total                = uct_iface->dev_addr_size +
                                  uct_iface->ep_addr_size;

    if (total > sizeof(addr->data)) {
        WARN("Address sizes are too big (%zu + %u > %zu)",
             uct_iface->dev_addr_size, uct_iface->ep_addr_size);
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
                                                uct_tl_resource_desc_t *tl)
{
    uct_iface_params_t params = {};
    ucs_status_t status;
    uct_iface_config_t *config;
    uct_iface_h iface;

    status = uct_md_iface_config_read(md, tl->tl_name, NULL, NULL, &config);
    if (status != UCS_OK) {
        WARN("Failed to read MD iface config for TL '%s': error %d",
             tl->tl_name, status);
        return NULL;
    }

    params.field_mask           = UCT_IFACE_PARAM_FIELD_OPEN_MODE   |
                                  UCT_IFACE_PARAM_FIELD_DEVICE      |
                                  UCT_IFACE_PARAM_FIELD_STATS_ROOT  |
                                  UCT_IFACE_PARAM_FIELD_RX_HEADROOM ;
    params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
    params.mode.device.tl_name  = tl->tl_name;
    params.mode.device.dev_name = tl->dev_name;
    params.stats_root           = NULL;
    params.rx_headroom          = 0;

    status = uct_iface_open(md, worker, &params, config, &iface);
    uct_config_release(config);
    if (status != UCS_OK) {
        WARN("Failed to open iface %s/%s: error %d", tl->tl_name, tl->dev_name, status);
        return NULL;
    }

    uct_iface_progress_enable(iface, UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    /* TODO: Add RMA/AM support checks */
    return iface;
}

static uct_iface_h nccl_uct_md_iface_open(uct_worker_h worker,
                                          uct_component_h comp,
                                          unsigned md_index,
                                          const char *md_name,
                                          const char *tl_name,
                                          const char *dev_name,
                                          uct_md_h *md_p)
{
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
        WARN("Failed to query resources MD[%d/%s]; error %d",
             md_index, md_name, status);
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

static void nccl_uct_ep_destroy(nccl_uct_ep_t *uct_ep)
{
    uct_ep_destroy(uct_ep->ep);
    free(uct_ep);
}

static nccl_uct_ep_t *nccl_uct_ep_create(nccl_uct_iface_t *uct_iface)
{
    nccl_uct_ep_t *uct_ep;
    ucs_status_t status;
    uct_ep_params_t ep_params;

    uct_ep = calloc(1, sizeof(*uct_ep) + uct_iface->ep_addr_size);
    if (uct_ep == NULL) {
        WARN("Failed to alloc EP memory");
        return NULL;
    }

    uct_ep->addr      = (uct_ep_addr_t *)uct_ep->data;
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
                                              nccl_uct_ep_addr_t *addr)
{
    ucs_status_t status = uct_ep_connect_to_ep(uct_ep->ep,
                                               nccl_uct_ep_addr_dev(addr),
                                               nccl_uct_ep_addr_ep(addr));
    if (status != UCS_OK) {
        WARN("Failed to connect to EP: error %d", status);
        return ncclSystemError;
    }

    return ncclSuccess;
}

static void nccl_uct_completion_callback(uct_completion_t *comp)
{
    assert(comp->count == 0);
}

static void nccl_uct_completion_init(uct_completion_t *completion, int count)
{
    completion->func   = nccl_uct_completion_callback;
    completion->count  = count;
    completion->status = UCS_OK;
}

static void nccl_uct_comm_rdesc_insert(nccl_uct_rdesc_t *rdesc)
{
    nccl_uct_comm_t *comm = rdesc->comm;

    if (comm->rdesc_list.tail != NULL) {
        rdesc->prev = comm->rdesc_list.tail;
        comm->rdesc_list.tail->next = rdesc;
    } else {
        assert(comm->rdesc_list.head == NULL);
        rdesc->prev = NULL;
        comm->rdesc_list.head = rdesc;
    }

    comm->rdesc_list.tail = rdesc;
}

static void nccl_uct_comm_rdesc_del(nccl_uct_rdesc_t *rdesc)
{
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

static nccl_uct_rdesc_t *
nccl_uct_comm_rdesc_get(nccl_uct_comm_t *comm)
{
    nccl_uct_rdesc_t *rdesc = comm->free_rdesc;

    if (rdesc == NULL) {
        rdesc = calloc(1, sizeof(*rdesc));
    } else {
        comm->free_rdesc = rdesc->next;
    }

    rdesc->next     = NULL;
    rdesc->comm     = comm;
    return rdesc;
}

static size_t nccl_uct_rdesc_size(int n)
{
    return n * sizeof(nccl_uct_chunk_t) + sizeof(nccl_uct_rdesc_hdr_t);
}

/* Prepare a receive descriptor from irecv()/iflush() side */
static void nccl_uct_rdesc_set(nccl_uct_rdesc_t *rdesc,
                               uint64_t id, int n, void **data,
                               int *sizes, int *tags,
                               nccl_uct_memh_t **uct_memh)
{
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
    nccl_uct_completion_init(&rdesc->completion, 1); /* wait for ATP or Flush */

    /* Zero (iflush) or One or many receive request are contained */
    for (i = 0; i < n; i++) {
        desc->chunk[i].tag     = tags[i];
        desc->chunk[i].size    = sizes[i];
        desc->chunk[i].data    = data[i];
        desc->chunk[i].matched = 0;

        desc->chunk[i].rkey = uct_memh[i]->bundle.rkey;
    }
}

static nccl_uct_req_t *nccl_uct_rdesc_get_req(nccl_uct_rdesc_t *rdesc, int i,
                                              int size)
{
    assert(i < NCCL_UCX_UCT_MAX_RECVS);

    rdesc->reqs[i].size  = size;
    rdesc->reqs[i].rdesc = rdesc;
    return &rdesc->reqs[i];
}

static void nccl_uct_comm_rdesc_put(nccl_uct_rdesc_t *rdesc)
{
    nccl_uct_comm_t *comm = rdesc->comm;

    assert(comm != NULL);

    rdesc->desc.id   = -1;
    rdesc->comm      = NULL;
    rdesc->next      = comm->free_rdesc;
    comm->free_rdesc = rdesc;
}

/* On receiver side, after ->irecv(), expect corresponding ATP */
static ucs_status_t nccl_uct_atp_callback(void *arg, void *data, size_t length,
                                          unsigned flags)
{
    nccl_uct_atp_t *atp = (nccl_uct_atp_t *)((uint8_t *)data + 8);

    assert(length == (sizeof(*atp) + 8));
    assert(*(nccl_uct_comm_t **)data == atp->rdesc->comm);
    assert(atp->id == atp->rdesc->desc.id);
    assert(atp->count == atp->rdesc->desc.count);
    assert(atp->rdesc->completion.count == 1);

    atp->rdesc->completion.count--;
    memcpy(atp->rdesc->sizes, atp->sizes, atp->count * sizeof(*atp->sizes));
    return UCS_OK;
}

/* On sender side, asynchronously receive rdesc/RTR, later used by ->isend() */
static ucs_status_t nccl_uct_rtr_callback(void *arg, void *data, size_t length,
                                          unsigned flags)
{
    nccl_uct_comm_t *comm      = *(nccl_uct_comm_t **)data;
    nccl_uct_rdesc_hdr_t *desc = (nccl_uct_rdesc_hdr_t *)((uint8_t *)data + 8);
    size_t size                = desc->size;
    nccl_uct_rdesc_t *rdesc;

    rdesc = nccl_uct_comm_rdesc_get(comm);
    if (rdesc == NULL) {
        WARN("Failed to get an rdesc in RTR callback");
        return UCS_ERR_NO_MEMORY; /* Cannot happend */
    }

    nccl_uct_comm_rdesc_insert(rdesc);

    assert((size + 8) == length);
    assert(size == nccl_uct_rdesc_size(desc->count));

    memcpy(&rdesc->desc, desc, size);
    rdesc->nccl_usage = desc->count;
    rdesc->send_atp   = desc->count + 1;
    nccl_uct_completion_init(&rdesc->completion, desc->count + 1);
    return UCS_OK;
}

static ncclResult_t nccl_uct_iface_set_handler(nccl_uct_iface_t *uct_iface,
                                               int id,
                                               uct_am_callback_t callback,
                                               nccl_uct_comm_t *comm)
{
    ucs_status_t status = uct_iface_set_am_handler(uct_iface->iface,
                                                   id, callback, comm, 0);
    if (status != UCS_OK) {
        WARN("Failed to get AM handler id=%d comm=%p", id, comm, status);
        return ncclInternalError;
    }

    return ncclSuccess;
}

static ncclResult_t nccl_uct_iface_set_rtr_mode(nccl_uct_iface_t *uct_iface,
                                                nccl_uct_comm_t *comm)
{
    NCCLCHECK(nccl_uct_iface_set_handler(uct_iface, NCCL_UCT_AM_RTR,
                                         nccl_uct_rtr_callback, NULL));
    NCCLCHECK(nccl_uct_iface_set_handler(uct_iface, NCCL_UCT_AM_ATP,
                                         nccl_uct_atp_callback, NULL));
    return ncclSuccess;
}

static void nccl_uct_iface_close(nccl_uct_iface_t *uct_iface)
{
    uct_iface_close(uct_iface->iface);
    uct_md_close(uct_iface->md);
    free(uct_iface->dev_addr);
    free(uct_iface->addr);
    free(uct_iface);
}

static nccl_uct_iface_t *nccl_uct_iface_open(nccl_uct_worker_t *uct_worker,
                                             const char *tl_name,
                                             const char *dev_name)
{
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
        status = uct_component_query(*comp, &comp_attr);
        if (status != UCS_OK) {
            WARN("Failed to query component: error %d", status);
            goto out;
        }

        comp_attr.field_mask   = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
        comp_attr.md_resources = alloca(sizeof(*comp_attr.md_resources) *
                                        comp_attr.md_resource_count);
        status = uct_component_query(*comp, &comp_attr);
        if (status != UCS_OK) {
            WARN("Failed to query component resources: error %d", status);
            goto out;
        }

        for (i = 0; i < comp_attr.md_resource_count; i++) {
            iface = nccl_uct_md_iface_open(worker, *comp, i,
                                           comp_attr.md_resources[i].md_name,
                                           tl_name, dev_name,
                                           &md);
            if (iface != NULL) {
                goto found;
            }
        }
    }

    if (iface == NULL) {
        WARN("Failed to open iface for tl_name=%s dev_name=%s",
             tl_name, dev_name);
        goto out;
    }

found:
    status = uct_iface_query(iface, &iface_attr);
    if (status != UCS_OK) {
        WARN("Failed to query iface for tl_name=%s dev_name=%s",
             tl_name, dev_name);
        goto fail;
    }

    if (!(iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP)) {
        WARN("Interface flag CONNECT_TO_EP is not set");
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
        WARN("Failed RTR does not fit in face AM short (%dB > %dB)",
             rtr_size, uct_iface->am_max_short);
        goto fail;
    }

    if (uct_iface->rkey_packed_size > sizeof(((nccl_uct_chunk_t *)0)->rkey)) {
        WARN("Interface rkey_packed_size %d too big",
             uct_iface->rkey_packed_size);
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
            WARN("Failed to query iface addr to tl_name=%s dev_name=%s",
                 tl_name, dev_name);
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

static ncclResult_t nccl_uct_init(ncclDebugLogger_t logFunction)
{
    return nccl_p2p_ib_init(&context.dev_count, ncclIbDevs, context.if_name,
                            &context.if_addr, NULL, logFunction);
}

static ncclResult_t nccl_uct_devices(int *ndev)
{
    *ndev = context.dev_count;
    return ncclSuccess;
}

static ncclResult_t nccl_uct_get_properties(int dev, ncclNetProperties_t* props)
{
    return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

static const char *nccl_dev_name(int dev)
{
    static __thread char buf[64];
    snprintf(buf, sizeof(buf), "%s:%d", ncclIbDevs[dev].devName,
             ncclIbDevs[dev].portNum);
    return buf;
}

static ncclResult_t nccl_uct_worker_create(nccl_uct_worker_t *w,
                                           nccl_uct_context_t *context,
                                           int dev)
{
    ucs_status_t status;

    status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK,
                                      &w->async);
    if (status != UCS_OK) {
        WARN("Failed to create UCT async context: dev=%d", dev);
        goto fail;
    }

    /* TODO: Fix thread mode */
    status = uct_worker_create(w->async, UCS_THREAD_MODE_SINGLE,
                               &w->worker);
    if (status != UCS_OK) {
        WARN("Failed to create UCT worker: dev=%d", dev);
        ucs_async_context_destroy(w->async);
        goto fail;
    }

    w->id.dev     = dev;
    w->id.thread  = pthread_self();
    w->context    = context;

    w->uct_iface = nccl_uct_iface_open(w, context->tl_name, nccl_dev_name(dev));
    if (w->uct_iface == NULL) {
        WARN("Failed to create UCT iface for worker: dev=%d", dev);
        uct_worker_destroy(w->worker);
        ucs_async_context_destroy(w->async);
        goto fail;
    }

    NCCLCHECK(nccl_uct_iface_set_rtr_mode(w->uct_iface, NULL));
    return ncclSuccess;

fail:
    return ncclSystemError;
}

static nccl_uct_worker_t *nccl_uct_worker_get(nccl_uct_context_t *context,
                                              int dev)
{
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

found:
    w->count++;
    w->next              = context->worker_list;
    context->worker_list = w;

out:
    pthread_mutex_unlock(&nccl_uct_lock);
    return w;
}

static ncclResult_t nccl_uct_listen(int dev, void *listen_handle,
                                    void **listen_comm)
{
    nccl_uct_listen_handle_t *handle = listen_handle;
    nccl_uct_listen_comm_t   *l_comm = calloc(1, sizeof(*l_comm));
    nccl_uct_comm_t *accept_comm;
    union ncclSocketAddress addr;

    if (l_comm == NULL) {
        WARN("Failed to alloc UCT listener(dev=%d)", dev);
        return ncclSystemError;
    }

    NCCL_STATIC_ASSERT(
                   sizeof(nccl_uct_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                   "UCT listen handle is too big");

    NCCLCHECK(ncclSocketInit(&l_comm->sock, &context.if_addr,
                             NCCL_UCT_LISTEN_HANDLE_MAGIC,
                             ncclSocketTypeNetIb, NULL, 1));
    NCCLCHECK(ncclSocketListen(&l_comm->sock));
    NCCLCHECK(ncclSocketGetAddr(&l_comm->sock, &addr));

    l_comm->uct_worker = nccl_uct_worker_get(&context, dev);
    if (l_comm->uct_worker == NULL) {
        WARN("Failed to create worker for listener dev=%d", dev);
        return ncclSystemError;
    }

    NCCLCHECK(ncclIbMalloc((void **)&accept_comm, sizeof(*accept_comm)));

    l_comm->comm       = accept_comm;
    l_comm->context    = &context;
    l_comm->dev        = dev;
    l_comm->id         = context.listener_count++;

    *listen_comm = l_comm;

    memset(handle, 0, sizeof(*handle));
    handle->magic         = NCCL_UCT_LISTEN_HANDLE_MAGIC;
    handle->listener.id   = l_comm->id;
    handle->listener.addr = addr;
    handle->comm          = accept_comm;

    WARN("Listening id=%d dev=%d comm=%p", l_comm->id, dev, handle->comm);
    return ncclSuccess;
}

static ncclResult_t nccl_uct_close_listen(void *listen_comm)
{
    nccl_uct_listen_comm_t *comm = listen_comm;

    if (comm) {
        NCCLCHECK(ncclSocketClose(&comm->sock));
        free(comm);
    }
    return ncclSuccess;
}

static ncclResult_t nccl_uct_comm_init(nccl_uct_comm_t *comm,
                                       nccl_uct_context_t *context,
                                       nccl_uct_worker_t *worker,
                                       int dev,
                                       const nccl_uct_comm_t *remote_comm)
{
    if (worker == NULL) {
        worker = nccl_uct_worker_get(context, dev);
    }

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

    return ncclSuccess;
}

/* TODO recheck if still works with actual iflush */
static ncclResult_t nccl_uct_comm_gpu_flush_init(nccl_uct_comm_t *comm)
{
    ucs_status_t status;

    comm->gpu_flush.enabled =
        (nccl_p2p_gdr_support(comm->dev) == ncclSuccess) ||
        (nccl_p2p_dmabuf_support(comm->dev) == ncclSuccess);

    if (!comm->gpu_flush.enabled) {
        return ncclSuccess;
    }

    comm->gpu_flush.uct_ep  = nccl_uct_ep_create(comm->uct_iface);
    if (comm->gpu_flush.uct_ep == NULL) {
        return ncclSystemError;
    }

    NCCLCHECK(nccl_uct_ep_addr_set(&comm->gpu_flush.addr, comm,
                                   comm->gpu_flush.uct_ep));
    NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->gpu_flush.uct_ep,
                                        &comm->gpu_flush.addr));

    /* Remove when replaced by bcopy */
    status = uct_md_mem_reg(comm->uct_iface->md,
                            (void *)comm->gpu_flush.mem,
                            sizeof(comm->gpu_flush.mem),
                            UCT_MD_MEM_ACCESS_ALL,
                            &comm->gpu_flush.memh);
    return status == UCS_OK? ncclSuccess : ncclSystemError;
}

static ncclResult_t nccl_uct_connect(int dev, void *listen_handle,
                                     void **send_comm,
                                     ncclNetDeviceHandle_t** sendDevComm)
{
    int ready                        = 0;
    nccl_uct_listen_handle_t *handle = listen_handle;
    nccl_uct_stage_t *stage          = &handle->stage;
    nccl_uct_comm_t *comm            = stage->comm;
    nccl_uct_comm_addr_t addr;

    *send_comm = NULL;

    switch (stage->state) {
    case NCCL_UCT_START:
        NCCLCHECK(ncclIbMalloc((void **)&comm, sizeof(*comm)));
        NCCLCHECK(nccl_uct_comm_init(comm, &context, NULL, dev, handle->comm));
        NCCLCHECK(ncclSocketInit(&comm->sock, &handle->listener.addr,
                                 handle->magic, ncclSocketTypeNetIb, NULL, 1));
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
        NCCLCHECK(ncclSocketSend(&comm->sock, &addr, sizeof(addr)));
        NCCLCHECK(ncclSocketSend(&comm->sock, &comm, sizeof(comm)));

        stage->offset = 0;
        stage->state  = NCCL_UCT_RECEIVE_ADDR;
        /* fallthrough */

    case NCCL_UCT_RECEIVE_ADDR:
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock,
                                     &comm->addr, sizeof(comm->addr),
                                     &stage->offset));
        if (stage->offset != sizeof(comm->addr)) {
            return ncclSuccess; /* In progress */
        }

        ready = 1;
        NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep, &comm->addr.rma));
        NCCLCHECK(ncclSocketSend(&comm->sock, &ready, sizeof(ready)));

        *send_comm = comm;
        stage->state = NCCL_UCT_DONE;
        WARN("Connected comm=%p remote_comm=%p listener_id=%d",
             comm, comm->remote_comm, handle->listener.id);
        break;

    default:
        WARN("UCT connnect for dev=%d using unsupported state %d",
             dev, stage->state);
        return ncclSystemError;
    }

    return ncclSuccess;
}

static ncclResult_t nccl_uct_accept(void *listen_comm, void **recv_comm,
                             ncclNetDeviceHandle_v7_t** recvDevComm)
{
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
        NCCLCHECK(ncclSocketSend(&comm->sock, &addr, sizeof(addr)));

        stage->offset = 0;
        stage->state = NCCL_UCT_RECEIVE_ADDR;
        /* fallthrough */

    case NCCL_UCT_RECEIVE_ADDR:
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock,
                                     &comm->addr, sizeof(comm->addr),
                                     &stage->offset));
        if (stage->offset != sizeof(comm->addr)) {
            return ncclSuccess;
        }

        NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep, &comm->addr.rma));

        stage->offset = 0;
        stage->state = NCCL_UCT_RECEIVE_COMM;
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
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock,
                                     &stage->ready, sizeof(stage->ready),
                                     &stage->offset));
        if (stage->offset != sizeof(ready)) {
            return ncclSuccess;
        }
        if (stage->ready != 1) {
            WARN("Accepted comm=%p invalid status (ready=%d)", stage->ready);
            return ncclSystemError;
        }

        *recv_comm   = comm;
        stage->state = NCCL_UCT_DONE;
        WARN("Accepted comm=%p remote_comm=%p listener_id=%d",
             comm, comm->remote_comm, l_comm->id);
        break;

    default:
        WARN("UCT accept for dev=%d using unsupported state %d",
             l_comm->dev, stage->state);
        return ncclSystemError;
    }

    return ncclSuccess;
}

static ncclResult_t nccl_uct_reg_mr(void *reg_comm, void *data, size_t size,
                                    int type, void **mhandle)
{
    nccl_uct_comm_t *comm = reg_comm;
    uct_component_h comp  = comm->uct_iface->comp;
    uct_md_h md           = comm->uct_iface->md;
    intptr_t addr         = (intptr_t)data;
    size_t rkey_size      = comm->uct_iface->rkey_packed_size;
    nccl_uct_memh_t *uct_memh;
    ucs_status_t status;

    NCCLCHECK(ncclIbMalloc((void **)&uct_memh, sizeof(*uct_memh) + rkey_size));
    uct_memh->comm = comm;

    /* Use integral pages */
    size += addr & (NCCL_UCT_REG_ALIGN - 1);
    size  = (size + NCCL_UCT_REG_ALIGN - 1) & ~(NCCL_UCT_REG_ALIGN - 1);
    addr &= ~(NCCL_UCT_REG_ALIGN - 1);

    /* Register memory */
    status = uct_md_mem_reg(md, (void *)addr, size, UCT_MD_MEM_ACCESS_ALL,
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
    status = uct_rkey_unpack(comp, uct_memh->rkey,
                             &uct_memh->bundle);
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
                                           void **mhandle)
{
    return nccl_uct_reg_mr(reg_comm, data, size, type, mhandle);
}

static ncclResult_t nccl_uct_dereg_mr(void *dereg_comm, void *mhandle)
{
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
        WARN("Failed to deregister memh %p on comm %p: error %d",
             uct_memh, comm, status);
        return ncclSystemError;
    }

    uct_memh->comm = NULL;
    free(uct_memh);
    return ncclSuccess;
}

static void nccl_uct_send_atp(nccl_uct_comm_t *comm,
                              nccl_uct_rdesc_t *rdesc)
{
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
        rdesc->completion.count--;
        rdesc->send_atp = 0;
    }
}

static ncclResult_t nccl_uct_send(nccl_uct_comm_t *comm, void *data, int size,
                                  nccl_uct_memh_t *uct_memh,
                                  nccl_uct_rdesc_t *rdesc, int i,
                                  void **request)
{
    ucs_status_t status;
    uct_iov_t iov;

    *request = NULL;

    /* Details for local data */
    iov.buffer = data;
    iov.length = size;
    iov.memh   = uct_memh->memh;
    iov.stride = iov.length;
    iov.count  = 1;

    assert(size <= rdesc->desc.chunk[i].size);

    status = uct_ep_put_zcopy(comm->uct_ep->ep, &iov, 1,
                              (uint64_t)rdesc->desc.chunk[i].data,
                              rdesc->desc.chunk[i].rkey,
                              &rdesc->completion);

    if (status == UCS_OK) {
        rdesc->completion.count--;
    } else if (status != UCS_INPROGRESS) {
        return ncclSuccess;
    }

    rdesc->desc.chunk[i].matched = 1;
    --rdesc->send_atp;

    if (rdesc->send_atp == 1) {
        nccl_uct_comm_rdesc_del(rdesc); /* all ->isend() were now matched */
        nccl_uct_send_atp(comm, rdesc);
    }

    *request = nccl_uct_rdesc_get_req(rdesc, i, size); /* NCCL request */
    return ncclSuccess;
}

static ncclResult_t nccl_uct_isend(void *send_comm, void *data, int size,
                                   int tag, void *mhandle, void **request)
{
    nccl_uct_comm_t *comm = send_comm;
    nccl_uct_rdesc_t *rdesc;
    int i;

    *request = NULL;

    for (rdesc = comm->rdesc_list.head; rdesc != NULL; rdesc = rdesc->next) {
        for (i = 0; i < rdesc->desc.count; i++) {
            if (rdesc->desc.chunk[i].matched ||
                (rdesc->desc.chunk[i].tag != tag)) {
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
                                   void **request)
{
    nccl_uct_comm_t *comm      = recv_comm;
    nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t **)mhandles;
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
        *request = nccl_uct_rdesc_get_req(rdesc, 0, NCCL_UCT_REQ_IRECV);
    }

    return ncclSuccess;
}

static ncclResult_t nccl_uct_iflush(void *recv_comm, int n, void **data,
                                    int *sizes, void **mhandle, void **request)
{
    int last                   = -1;
    nccl_uct_comm_t *comm      = recv_comm;
    nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t **)mhandle;
    nccl_uct_rdesc_t *rdesc;
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

    rdesc = nccl_uct_comm_rdesc_get(comm);
    if (rdesc == NULL) {
        return ncclInternalError;
    }

    nccl_uct_rdesc_set(rdesc, ~0, 0, NULL, NULL, NULL, NULL);

    iov.buffer  = comm->gpu_flush.mem;
    iov.length  = sizeof(comm->gpu_flush.mem);
    iov.memh    = comm->gpu_flush.memh;
    iov.stride  = iov.length;
    iov.count   = 1;

    status = uct_ep_get_zcopy(comm->gpu_flush.uct_ep->ep,
                              &iov, 1, 
                              (uint64_t)data[last], uct_memh[last]->bundle.rkey,
                              &rdesc->completion);
    if (status == UCS_OK) {
        *request = NULL;
        nccl_uct_comm_rdesc_put(rdesc);
    } else if (status == UCS_INPROGRESS) {
        *request = nccl_uct_rdesc_get_req(rdesc, 0, NCCL_UCT_REQ_IFLUSH);
    } else {
        WARN("Failed to flush local ep comm=%p", comm);
        return ncclInternalError;
    }

    return ncclSuccess;
}

static ncclResult_t nccl_uct_test(void *request, int *done, int *sizes)
{
    nccl_uct_req_t *req     = request;
    nccl_uct_rdesc_t *rdesc = req->rdesc;
    nccl_uct_comm_t *comm   = rdesc->comm;

    uct_worker_progress(comm->uct_worker->worker);

    if (rdesc->send_atp == 1) {
        /* unlikely */
        nccl_uct_send_atp(comm, rdesc);
    }

    if (rdesc->completion.count > 0) {
        *done = 0;
        return ncclSuccess;
    }

    assert(rdesc->send_atp == 0);
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
        assert(rdesc->nccl_usage == 0);
        nccl_uct_comm_rdesc_put(rdesc);
    }

    return ncclSuccess;
}

static void nccl_uct_worker_destroy(nccl_uct_worker_t *w)
{
    nccl_uct_iface_close(w->uct_iface);
    uct_worker_destroy(w->worker);
    ucs_async_context_destroy(w->async);
    free(w);
}

static void nccl_uct_worker_put(nccl_uct_worker_t *worker)
{
    nccl_uct_worker_t **wp;

    pthread_mutex_lock(&nccl_uct_lock);
    if (--worker->count < 1) {
        assert(worker->count == 0);
        for (wp = &worker->context->worker_list; *wp != NULL;
             wp = &(*wp)->next) {
            if (*wp == worker) {
                *wp = worker->next;
                nccl_uct_worker_destroy(worker);
                break;
            }
        }
    }
    pthread_mutex_unlock(&nccl_uct_lock);
}

static ncclResult_t nccl_uct_close(void *close_comm)
{
    nccl_uct_comm_t *comm = close_comm;
    nccl_uct_rdesc_t *rdesc;

    nccl_uct_ep_destroy(comm->uct_ep);
    if (comm->gpu_flush.uct_ep != NULL) {
        nccl_uct_ep_destroy(comm->gpu_flush.uct_ep);
        (void)uct_md_mem_dereg(comm->uct_iface->md,
                               comm->gpu_flush.memh);
    }
    nccl_uct_worker_put(comm->uct_worker);

    while ((rdesc = comm->free_rdesc) != NULL) {
        comm->free_rdesc = rdesc->next;
        free(rdesc);
    }

    assert(comm->rdesc_list.head == NULL);
    assert(comm->rdesc_list.tail == NULL);
    free(comm);

    return ncclSuccess;
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
    .regMrDmaBuf   = nccl_uct_reg_mr_dmabuf, /* TODO: Test perf when fd != -1 */
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
