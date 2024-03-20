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
    NCCL_UCT_SEND_ADDR
} nccl_uct_state_t;

typedef struct nccl_uct_worker {
    struct nccl_uct_worker *next;
    struct {
        pthread_t           thread; /* TODO: double check that part */
        int                 dev;
    } id;

    ucs_async_context_t     *uct_async;
    uct_worker_h            uct_worker;
    struct nccl_uct_context *context;
} nccl_uct_worker_t;

typedef struct {
    struct ncclSocket       sock;
    struct nccl_uct_context *context;
    nccl_uct_worker_t       *worker;
} nccl_uct_comm_t;

typedef struct {
    nccl_uct_state_t state;
    nccl_uct_comm_t *comm;
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
    int               dev;
    uint32_t          id;
    struct ncclSocket sock;
} nccl_uct_listen_comm_t;

typedef struct {
} nccl_uct_rx_desc_t;

/* Global state of the plugin */
typedef struct nccl_uct_context {
    int                     dev_count; /* How many available devices */
    char                    if_name[MAX_IF_NAME_SIZE];
    union ncclSocketAddress if_addr;

    uint32_t                listener_id; /* Listener ID allocation */

    nccl_uct_worker_t       *worker_list; /* List of instanciated workers */
} nccl_uct_context_t;

#define NCCL_UCT_LISTEN_HANDLE_MAGIC 0x43cf19ed91abdb85


static pthread_mutex_t nccl_uct_lock = PTHREAD_MUTEX_INITIALIZER;

static nccl_uct_context_t context;


static uct_iface_h nccl_uct_iface_open(uct_worker_h worker,
                                       uct_md_h md, uct_tl_resource_desc_t *tl)
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
                                          const char *dev_name)
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
        if (strcmp(dev_name, tls[i].dev_name) ||
            strcmp(tl_name, tls[i].tl_name)) {
            continue;
        }

        iface = nccl_uct_iface_open(worker, md, &tls[i]);
        break;
    }
    uct_release_tl_resource_list(tls);

out:
    uct_md_close(md);
    return iface;
}

static uct_iface_h nccl_uct_device_open(uct_worker_h worker,
                                        const char *tl_name,
                                        const char *dev_name)
{
    uct_iface_h iface = NULL;
    uct_component_h *comps, *comp;
    unsigned comps_count, i;
    ucs_status_t status;
    uct_component_attr_t comp_attr;

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

        for (i = 0; i < comp_attr.md_resource_count; i++) {
            iface = nccl_uct_md_iface_open(worker, *comp, i,
                                           comp_attr.md_resources[i].md_name,
                                           tl_name, dev_name);
            if (iface != NULL) {
                break;
            }
        }
    }

out:
    uct_release_component_list(comps);
    return iface;
}

static ncclResult_t nccl_uct_init(ncclDebugLogger_t logFunction)
{
    uct_iface_h iface = nccl_uct_device_open(NULL, "rc_x", "mlx5_0:1");
    (void)iface;
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

    comm->dev    = dev;
    comm->id     = context.listener_id++;
    *listen_comm = comm;

    memset(handle, 0, sizeof(*handle));
    handle->magic         = NCCL_UCT_LISTEN_HANDLE_MAGIC;
    handle->listener.id   = comm->id;
    handle->listener.addr = addr;

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

static ncclResult_t nccl_uct_worker_init(nccl_uct_worker_t *w,
                                         nccl_uct_context_t *context,
                                         int dev)
{
    ucs_status_t status;

    /* Create UCT objects */
    status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK,
                                      &w->uct_async);
    if (status != UCS_OK) {
        WARN("Failed to create UCT async context: dev=%d", dev);
        goto fail;
    }

    status = uct_worker_create(w->uct_async, UCS_THREAD_MODE_MULTI,
                               &w->uct_worker);
    if (status != UCS_OK) {
        WARN("Failed to create UCT worker: dev=%d", dev);
        ucs_async_context_destroy(w->uct_async);
        goto fail;
    }

    /* Initialize */
    w->id.dev     = dev;
    w->id.thread  = pthread_self();
    w->context = context;

    /* Add to worker list */
    w->next              = context->worker_list;
    context->worker_list = w;
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
        if ((w->id.dev == dev) && (w->id.thread == pthread_self())) {
            break;
        }
    }

    if (w == NULL) {
        w = calloc(1, sizeof(*w));
        if (w == NULL) {
            WARN("Failed worker allocation: dev=%d", dev);
            goto out;
        }

        if (nccl_uct_worker_init(w, context, dev) != ncclSuccess) {
            free(w);
            w = NULL;
            goto out;
        }
    }

out:
    pthread_mutex_unlock(&nccl_uct_lock);
    return w;
}

static ncclResult_t nccl_uct_comm_init(nccl_uct_comm_t *comm,
                                       nccl_uct_context_t *context,
                                       int dev)
{
    comm->context = context;
    comm->worker  = nccl_uct_worker_get(context, dev);
    /* TODO: Create iface */
    return (comm->worker == NULL) ? ncclSystemError : ncclSuccess;
}

static ncclResult_t nccl_uct_connect(int dev, void *listen_handle,
                                     void **send_comm,
                                     ncclNetDeviceHandle_t** sendDevComm)
{
    nccl_uct_listen_handle_t *handle = listen_handle;
    nccl_uct_stage_t *stage          = &handle->stage;
    int ready;

    *send_comm = NULL;

    switch (stage->state) {
    case NCCL_UCT_START:
        NCCLCHECK(ncclIbMalloc((void **)&stage->comm, sizeof(stage->comm)));
        NCCLCHECK(ncclSocketInit(&stage->comm->sock, &handle->listener.addr,
                                 handle->magic, ncclSocketTypeNetIb, NULL, 1));
        NCCLCHECK(ncclSocketConnect(&stage->comm->sock));
        stage->state = NCCL_UCT_CONNECT;
        /* fallthrough */
    case NCCL_UCT_CONNECT:
        NCCLCHECK(ncclSocketReady(&stage->comm->sock, &ready));
        if (!ready) {
            return ncclSuccess;
        }

        NCCLCHECK(nccl_uct_comm_init(stage->comm, &context, dev));
        stage->state = NCCL_UCT_SEND_ADDR;
        /* fallthrough */
    case NCCL_UCT_SEND_ADDR:
        break;
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
    .accept = NULL,
    .regMr = NULL,
    .regMrDmaBuf = NULL,
    .deregMr = NULL,
    .isend = NULL,
    .irecv = NULL,
    .iflush = NULL,
    .test = NULL,
    .closeSend = NULL,
    .closeRecv = NULL,
    .closeListen   = nccl_uct_close_listen,
    NULL /* getDeviceMr */,
    NULL /* irecvConsumed */
};
