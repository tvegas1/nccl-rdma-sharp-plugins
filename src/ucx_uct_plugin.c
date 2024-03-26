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

struct nccl_uct_context;

typedef enum {
    NCCL_UCT_START = 0,
    NCCL_UCT_CONNECT,
    NCCL_UCT_ACCEPT,
    NCCL_UCT_RECEIVE_ADDR,
    NCCL_UCT_DONE
} nccl_uct_state_t;

/* UCT EP address to exchange and connect to */
typedef struct {
    uint8_t dev_addr_size;
    uint8_t ep_addr_size;
    uint8_t data[64]; /* TODO: Don't hardcode value, fix align */
} nccl_uct_ep_addr_t;

typedef struct {
    uct_iface_h             iface;
    uct_md_h                md;
    void                    *addr;
    size_t                  addr_size;
    void                    *dev_addr;
    size_t                  dev_addr_size;
    size_t                  ep_addr_size;
} nccl_uct_iface_t;

typedef struct nccl_uct_worker {
    struct nccl_uct_worker *next;
    struct {
        pthread_t           thread; /* TODO: double check that part */
        int                 dev;
    } id;

    ucs_async_context_t     *async;
    uct_worker_h            worker;
    struct nccl_uct_context *context;
    nccl_uct_iface_t        *uct_iface;
} nccl_uct_worker_t;

typedef struct {
    uct_ep_h                ep;
    uct_ep_addr_t           *addr;
    size_t                  addr_size;
    nccl_uct_iface_t        *uct_iface;
    uint8_t                 data[];
} nccl_uct_ep_t;

typedef struct {
    struct ncclSocket       sock;
    struct nccl_uct_context *context;
    nccl_uct_worker_t       *uct_worker;

    nccl_uct_iface_t        *uct_iface;
    nccl_uct_ep_t           *uct_ep;
    nccl_uct_ep_addr_t      addr; /* remote addr */
} nccl_uct_comm_t;

typedef struct {
    nccl_uct_state_t state;
    nccl_uct_comm_t  *comm;
    int              offset;
} nccl_uct_stage_t;

typedef uint64_t nccl_uct_tag_t;

/* Passed around by NCCL */
typedef struct {
    uint64_t                    magic;
    struct {
        union ncclSocketAddress addr;
        uint32_t                id;
    } listener;
    nccl_uct_stage_t            stage;
} nccl_uct_listen_handle_t;

/* Communicator while listening to remote ranks */
typedef struct {
    int                     dev;
    uint32_t                id;
    struct ncclSocket       sock;
    nccl_uct_tag_t          tag;
    nccl_uct_worker_t       *uct_worker;
    struct nccl_uct_context *context;

    nccl_uct_stage_t  stage;
} nccl_uct_listen_comm_t;

typedef struct {
} nccl_uct_rx_desc_t;

/* Global state of the plugin */
typedef struct nccl_uct_context {
    const char              *tl_name;

    int                     dev_count; /* How many available devices */
    char                    if_name[MAX_IF_NAME_SIZE];
    union ncclSocketAddress if_addr;

    uint32_t                listener_id; /* Listener ID allocation */

    nccl_uct_worker_t       *worker_list; /* List of instanciated workers */

    nccl_uct_tag_t          tag[MAX_IB_DEVS];
} nccl_uct_context_t;

#define NCCL_UCT_LISTEN_HANDLE_MAGIC 0x43cf19ed91abdb85

#define NCCL_UCT_TAG_SHIFT 0x100

static pthread_mutex_t nccl_uct_lock = PTHREAD_MUTEX_INITIALIZER;

static nccl_uct_context_t context = {
    .tl_name   = "rc_mlx5",
    .dev_count = -1
};

static void nccl_uct_context_init(nccl_uct_context_t *context)
{
    for (int i = 0; i < MAX_IB_DEVS; i++) {
        context->tag[i] = i;
    }
}

static nccl_uct_tag_t nccl_uct_tag_get(nccl_uct_context_t *context, int dev)
{
    context->tag[dev] += NCCL_UCT_TAG_SHIFT;
    return context->tag[dev];
}

static const uct_device_addr_t *nccl_uct_ep_addr_dev(nccl_uct_ep_addr_t *addr)
{
    return (uct_device_addr_t *)addr->data;
}

static const uct_ep_addr_t *nccl_uct_ep_addr_ep(nccl_uct_ep_addr_t *addr)
{
    return (uct_ep_addr_t *)(addr->data + addr->dev_addr_size);
}

static ncclResult_t nccl_uct_ep_addr_set(nccl_uct_ep_addr_t *addr,
                                         const nccl_uct_comm_t *comm)
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
    memcpy(addr->data + addr->dev_addr_size, comm->uct_ep->addr,
           uct_iface->ep_addr_size);
    return ncclSuccess;
}

static uct_iface_h nccl_uct_resource_iface_open(uct_worker_h worker,
                                                uct_md_h md,
                                                uct_tl_resource_desc_t *tl)
{
    ucs_status_t status;
    uct_iface_config_t *config;
    uct_iface_h iface;
    uct_iface_params_t params;

    status = uct_md_iface_config_read(md, tl->tl_name, NULL, NULL, &config);
    if (status != UCS_OK) {
        WARN("Failed to read MD iface config for TL '%s': error %d",
             tl->tl_name, status);
        return NULL;
    }

    params.field_mask           = UCT_IFACE_PARAM_FIELD_OPEN_MODE   |
                                  UCT_IFACE_PARAM_FIELD_DEVICE      |
                                  UCT_IFACE_PARAM_FIELD_STATS_ROOT  |
                                  UCT_IFACE_PARAM_FIELD_RX_HEADROOM |
                                  UCT_IFACE_PARAM_FIELD_CPU_MASK;
    params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
    params.mode.device.tl_name  = tl->tl_name;
    params.mode.device.dev_name = tl->dev_name;
    params.stats_root           = NULL;
    params.rx_headroom          = sizeof(nccl_uct_rx_desc_t);

    status = uct_iface_open(md, worker, &params, config, &iface);
    uct_config_release(config);
    if (status != UCS_OK) {
        WARN("Failed to open iface %s/%s: error %d", tl->tl_name, tl->dev_name);
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

    uct_ep->addr = (uct_ep_addr_t *)uct_ep->data;

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
        free(uct_ep);
        return NULL;
    }

    return uct_ep;
}

static nccl_uct_iface_t *nccl_uct_iface_open(uct_worker_h worker,
                                             const char *tl_name,
                                             const char *dev_name)
{
    nccl_uct_iface_t *uct_iface = NULL;
    uct_iface_h iface          = NULL;
    uct_component_h *comps, *comp;
    unsigned comps_count, i;
    ucs_status_t status;
    uct_component_attr_t comp_attr;
    uct_iface_attr_t iface_attr;
    uct_md_h md;

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

    uct_iface = calloc(1, sizeof(*uct_iface));
    if (uct_iface == NULL) {
        WARN("Failed to alloc uct iface structure");
        goto fail;
    }

    uct_iface->ep_addr_size = iface_attr.ep_addr_len;
    uct_iface->md           = md;

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

    WARN("IFACE %p dlen %zu iface len %zu ep len %zu", iface,
         uct_iface->dev_addr_size, uct_iface->addr_size, uct_iface->ep_addr_size);
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
    sleep (10);
    nccl_uct_context_init(&context);
    return nccl_p2p_ib_init(&context.dev_count, ncclIbDevs, context.if_name,
                            &context.if_addr, NULL, logFunction);
}

static ncclResult_t nccl_uct_devices(int *ndev) {
    *ndev = context.dev_count;
    return ncclSuccess;
}

static ncclResult_t nccl_uct_get_properties(int dev, ncclNetProperties_t* props)
{
    return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

static ncclResult_t nccl_uct_worker_create(nccl_uct_worker_t *w,
                                           nccl_uct_context_t *context,
                                           int dev)
{
    ucs_status_t status;

    /* Create UCT objects */
    status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK,
                                      &w->async);
    if (status != UCS_OK) {
        WARN("Failed to create UCT async context: dev=%d", dev);
        goto fail;
    }

    status = uct_worker_create(w->async, UCS_THREAD_MODE_SERIALIZED,
                               &w->worker);
    if (status != UCS_OK) {
        WARN("Failed to create UCT worker: dev=%d", dev);
        ucs_async_context_destroy(w->async);
        goto fail;
    }

    /* Initialize */
    w->id.dev     = dev;
    w->id.thread  = pthread_self();
    w->context = context;

    return ncclSuccess;

fail:
    return ncclSystemError;
}

static const char *nccl_dev_name(int dev)
{
    static __thread char buf[64];
    snprintf(buf, sizeof(buf), "%s:%d", ncclIbDevs[dev].devName,
             ncclIbDevs[dev].portNum);
    return buf;
}

static nccl_uct_worker_t *nccl_uct_worker_get(nccl_uct_context_t *context,
                                              int dev)
{
    nccl_uct_worker_t *w;

    pthread_mutex_lock(&nccl_uct_lock);

    for (w = context->worker_list; w != NULL; w = w->next) {
        if ((w->id.dev == dev) && (w->id.thread == pthread_self())) {
            goto out;
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

    w->uct_iface = nccl_uct_iface_open(w->worker, context->tl_name,
                            nccl_dev_name(dev));
    if (w->uct_iface == NULL) {
        w = NULL; /* TODO fix leak */
        goto out;
    }

    /* Add to worker list */
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
    nccl_uct_listen_comm_t   *comm   = calloc(1, sizeof(*comm));
    union ncclSocketAddress addr;

    if (comm == NULL) {
        WARN("Failed to alloc UCT listener(dev=%d)", dev);
        return ncclSystemError;
    }

    NCCL_STATIC_ASSERT(
                       sizeof(nccl_uct_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                       "UCT listen handle is too big");

    NCCLCHECK(ncclSocketInit(&comm->sock, &context.if_addr,
                             NCCL_UCT_LISTEN_HANDLE_MAGIC,
                             ncclSocketTypeNetIb, NULL, 1));
    NCCLCHECK(ncclSocketListen(&comm->sock));
    NCCLCHECK(ncclSocketGetAddr(&comm->sock, &addr));

    comm->uct_worker = nccl_uct_worker_get(&context, dev);
    if (comm->uct_worker == NULL) {
        WARN("Failed to create worker for listener dev=%d", dev);
        return ncclSystemError;
    }

    comm->context    = &context;
    comm->dev        = dev;
    comm->id         = context.listener_id++;
    comm->tag        = nccl_uct_tag_get(&context, dev);

    *listen_comm = comm;

    memset(handle, 0, sizeof(*handle));
    handle->magic         = NCCL_UCT_LISTEN_HANDLE_MAGIC;
    handle->listener.id   = comm->id;
    handle->listener.addr = addr;

    WARN("Listen dev=%d ok", dev);
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
                                       int dev)
{
    comm->context    = context;
    comm->uct_worker = nccl_uct_worker_get(context, dev);
    if (comm->uct_worker == NULL) {
        return ncclSystemError;
    }

    comm->uct_iface  = comm->uct_worker->uct_iface;
    comm->uct_ep     = nccl_uct_ep_create(comm->uct_iface);
    if (comm->uct_ep == NULL) {
        return ncclSystemError;
    }

    return ncclSuccess;
}

static ncclResult_t nccl_uct_ep_connect_to_ep(nccl_uct_ep_t *uct_ep,
                                              nccl_uct_ep_addr_t *addr)
{
    ucs_status_t status = uct_ep_connect_to_ep(uct_ep->ep,
                                               nccl_uct_ep_addr_dev(addr),
                                               nccl_uct_ep_addr_ep(addr));
    if (status != UCS_OK) {
        WARN("Accept(dev=%d): failed to connect: error %d", status);
        return ncclSystemError;
    }

    return ncclSuccess;
}

static ncclResult_t nccl_uct_connect(int dev, void *listen_handle,
                                     void **send_comm,
                                     ncclNetDeviceHandle_t** sendDevComm)
{
    nccl_uct_listen_handle_t *handle = listen_handle;
    nccl_uct_stage_t *stage          = &handle->stage;
    nccl_uct_comm_t *comm            = stage->comm;
    nccl_uct_ep_addr_t addr;
    int ready;

    *send_comm = NULL;

    switch (stage->state) {
    case NCCL_UCT_START:
        NCCLCHECK(ncclIbMalloc((void **)&comm, sizeof(*comm)));
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

        NCCLCHECK(nccl_uct_comm_init(comm, &context, dev));
        stage->state = NCCL_UCT_RECEIVE_ADDR;
        NCCLCHECK(nccl_uct_ep_addr_set(&addr, comm));
        /* TODO: Add EP addresses for multiple QPs */
        NCCLCHECK(ncclSocketSend(&comm->sock, &addr, sizeof(addr)));

        WARN("connect for dev=%d w=%p i=%p ep=%p",
             dev, comm->uct_worker, comm->uct_iface, comm->uct_ep);
        /* fallthrough */
    case NCCL_UCT_RECEIVE_ADDR:
        /* TODO: Merge connect and accept codes */
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock,
                                     &comm->addr, sizeof(comm->addr),
                                     &stage->offset));
        if (stage->offset != sizeof(comm->addr)) {
            return ncclSuccess;
        }

        NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep, &comm->addr));
        WARN("connect rx'd");
        stage->state = NCCL_UCT_DONE;
        *send_comm = comm;
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
    nccl_uct_ep_addr_t addr;
    int ready;

    *recv_comm = NULL;

    switch (stage->state) {
    case NCCL_UCT_START:
        NCCLCHECK(ncclIbMalloc((void **)&comm, sizeof(*comm)));
        stage->comm  = comm;
        stage->state = NCCL_UCT_ACCEPT;
        NCCLCHECK(ncclSocketInit(&comm->sock, NULL, NCCL_SOCKET_MAGIC,
                                 ncclSocketTypeUnknown, NULL, 0));
        NCCLCHECK(ncclSocketAccept(&comm->sock, &l_comm->sock));
        /* fallthrough */
    case NCCL_UCT_ACCEPT:
        NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
        if (!ready) return ncclSuccess;

        comm->context    = l_comm->context;
        comm->uct_worker = l_comm->uct_worker;
        comm->uct_iface  = comm->uct_worker->uct_iface;
        comm->uct_ep     = nccl_uct_ep_create(comm->uct_iface);
        if (comm->uct_ep == NULL) {
            return ncclSystemError;
        }

        stage->state = NCCL_UCT_RECEIVE_ADDR;
        NCCLCHECK(nccl_uct_ep_addr_set(&addr, comm));
        NCCLCHECK(ncclSocketSend(&comm->sock, &addr, sizeof(addr)));

        WARN("accepted for dev=%d w=%p i=%p ep=%p",
             l_comm->dev, comm->uct_worker, comm->uct_iface, comm->uct_ep);

        /* fallthrough */
    case NCCL_UCT_RECEIVE_ADDR:
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock,
                                     &comm->addr, sizeof(comm->addr),
                                     &stage->offset));
        if (stage->offset != sizeof(comm->addr)) {
            return ncclSuccess;
        }

        NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep, &comm->addr));
        WARN("accept rx'd");
        stage->state = NCCL_UCT_DONE;
        *recv_comm = comm;
        break;

    default:
        WARN("UCT accept for dev=%d using unsupported state %d",
             l_comm->dev, stage->state);
        return ncclSystemError;
    }

    return ncclSuccess;
}

/* Memory registration handle in NCCL UCT plugin */
typedef struct {
    uct_mem_h memh;
    nccl_uct_comm_t *comm;
} nccl_uct_memh_t;

#define NCCL_UCT_REG_ALIGN 4096

static ncclResult_t nccl_uct_reg_mr(void *reg_comm, void *data, size_t size,
                                    int type, void **mhandle)
{
    nccl_uct_comm_t *comm = reg_comm;
    uct_md_h md           = comm->uct_iface->md;
    intptr_t addr         = (intptr_t)data;
    nccl_uct_memh_t *uct_memh;
    ucs_status_t status;

    NCCLCHECK(ncclIbMalloc((void **)&uct_memh, sizeof(*uct_memh)));
    uct_memh->comm = comm;

    /* Use integral pages */
    size += addr & (NCCL_UCT_REG_ALIGN - 1);
    size  = (size + NCCL_UCT_REG_ALIGN - 1) & ~(NCCL_UCT_REG_ALIGN - 1);
    addr &= ~(NCCL_UCT_REG_ALIGN - 1);

    status = uct_md_mem_reg(md, (void *)addr, size, UCT_MD_MEM_ACCESS_RMA,
                            &uct_memh->memh);
    if (status != UCS_OK) {
        WARN("Failed to register %p/%zu on comm %p", addr, size, comm);
        return ncclSystemError;
    }

    *mhandle = uct_memh;
    return ncclSuccess;
}

static ncclResult_t nccl_uct_reg_mr_dmabuf(void *reg_comm, void *data,
                                           size_t size, int type,
                                           uint64_t offset, int fd,
                                           void **mhandle)
{
    /* TODO: Use mem reg v2 with DMABUF flag? */
    return nccl_uct_reg_mr(reg_comm, data, size, type, mhandle);
}

static ncclResult_t nccl_uct_dereg_mr(void *dereg_comm, void *mhandle)
{
    nccl_uct_comm_t *comm     = dereg_comm;
    nccl_uct_memh_t *uct_memh = mhandle;
    ucs_status_t status;

    assert(uct_memh->memh != UCT_MEM_HANDLE_NULL);
    assert(uct_memh->comm == comm);

    status = uct_md_mem_dereg(comm->uct_iface->md, uct_memh->memh);
    if (status != UCS_OK) {
        WARN("Failed to deregister memh %p on comm %p: error %d",
             uct_memh, comm, status);
        return ncclSystemError;
    }

    return ncclSuccess;
}

static ncclResult_t nccl_ucx_irecv(void *recv_comm, int n, void **data,
                                   int *sizes, int *tags, void **mhandle,
                                   void **request)
{
    ucs_status_t status;

    if (n > NCCL_NET_IB_MAX_RECVS) {
        WARN("uct_irecv failed: n %d > %d", n, NCCL_NET_IB_MAX_RECVS);
        return ncclInternalError;
    }

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
    .regMrDmaBuf   = nccl_uct_reg_mr_dmabuf,
    .deregMr       = nccl_uct_dereg_mr,
    .isend = NULL,
    .irecv         = nccl_uct_irecv,
    .iflush = NULL,
    .test = NULL,
    .closeSend = NULL,
    .closeRecv = NULL,
    .closeListen   = nccl_uct_close_listen,
    NULL /* getDeviceMr */,
    NULL /* irecvConsumed */
};
