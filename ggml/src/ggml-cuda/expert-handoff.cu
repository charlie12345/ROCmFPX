// ROCmFPX [TAG_EXPERT_HANDOFF] expert handoff, the device side (protocol in ggml-expert-handoff.h).
// send: copy the routed experts' input and selection into the mailbox in host memory, then bump seq_send with a
//       system-scope release, so the host pool can start on its experts while this stream computes the cached ones.
// join: wait (bounded) until the host answered that send, then add its sum to ours. Both are plain kernels on the
//       stream, so they are captured into graphs like any other node.

#include "expert-handoff.cuh"
#include "ggml-expert-handoff.h"

#include <mutex>

struct eh_offsets {
    size_t x, ids, slots, weights, y;
};

static eh_offsets eh_offsets_of(const ggml_expert_handoff_dims d) {
    return { ggml_expert_handoff_off_x(d), ggml_expert_handoff_off_ids(d), ggml_expert_handoff_off_slots(d),
             ggml_expert_handoff_off_weights(d), ggml_expert_handoff_off_y(d) };
}

static __device__ __forceinline__ uint32_t eh_dev_load(const uint32_t * p) {
#if defined(GGML_USE_HIP)
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    const uint32_t v = *(const volatile uint32_t *) p;
    __threadfence_system();
    return v;
#endif
}

static __device__ __forceinline__ void eh_dev_store(uint32_t * p, const uint32_t v) {
#if defined(GGML_USE_HIP)
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    __threadfence_system();
    *(volatile uint32_t *) p = v;
#endif
}

static __device__ __forceinline__ uint64_t eh_dev_ticks() {
#if defined(GGML_USE_HIP)
    return wall_clock64();
#elif defined(GGML_USE_MUSA)
    return clock64();
#else
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

static __device__ __forceinline__ void eh_dev_pause() {
#if defined(GGML_USE_HIP)
    __builtin_amdgcn_s_sleep(2);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __nanosleep(200);
#endif
}

// one block: count the misses (flags[layer], read by the join) and the routing (counts); with misses, copy the
// selection into the mailbox, every thread fences its own writes and thread 0 rings, so the host sees all of them
static __global__ void k_expert_handoff_send(
        const float * __restrict__ x, const int32_t * __restrict__ ids, const int32_t * __restrict__ slot_map, const int n_map,
        const float * __restrict__ w, int32_t * __restrict__ slots_out, uint8_t * mb, const eh_offsets off,
        const int n_embd, const int n_tokens, const int n_used, const int layer, const int max_embd, const int max_used,
        const int miss_slot, int32_t * flags, int32_t * counts, const int counts_stride) {
    __shared__ int n_miss;
    if (threadIdx.x == 0) {
        n_miss = 0;
    }
    __syncthreads();
    // each thread looks up the slots of its selections (the second loop below takes the same ones)
    for (int i = threadIdx.x; i < n_tokens*n_used; i += blockDim.x) {
        const int32_t e = ids[i];
        const int32_t s = e >= 0 && e < n_map ? slot_map[e] : miss_slot;
        slots_out[i] = s;
        if (s == miss_slot) {
            atomicAdd(&n_miss, 1);
        }
        if (counts != nullptr && e >= 0 && e < counts_stride) {
            atomicAdd(&counts[layer*counts_stride + e], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        flags[layer] = n_miss;
    }
    if (n_miss == 0) {
        return;
    }

    float   * mx = (float   *) (mb + off.x);
    int32_t * mi = (int32_t *) (mb + off.ids);
    int32_t * ms = (int32_t *) (mb + off.slots);
    float   * mw = (float   *) (mb + off.weights);

    for (int i = threadIdx.x; i < n_tokens*n_embd; i += blockDim.x) {
        const int t = i / n_embd;
        const int c = i - t*n_embd;
        mx[t*max_embd + c] = x[i];
    }
    for (int i = threadIdx.x; i < n_tokens*n_used; i += blockDim.x) {
        const int t = i / n_used;
        const int k = i - t*n_used;
        mi[t*max_used + k] = ids[i];
        ms[t*max_used + k] = slots_out[i];
        mw[t*max_used + k] = w[i];
    }

    ggml_expert_handoff_header * hdr = (ggml_expert_handoff_header *) mb;
    if (threadIdx.x == 0) {
        hdr->layer    = layer;
        hdr->n_tokens = n_tokens;
        hdr->n_embd   = n_embd;
        hdr->n_used   = n_used;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        eh_dev_store(&hdr->seq_send, eh_dev_load(&hdr->seq_send) + 1);
    }
}

static __global__ void k_expert_handoff_join(
        const float * __restrict__ y, float * __restrict__ dst, uint8_t * mb, const size_t off_y,
        const int n_embd, const int n, const int max_embd, const uint64_t timeout_ticks,
        const int32_t * __restrict__ flags, const int layer) {
    if (flags[layer] == 0) {
        // nothing was sent: no host experts to add
        for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
            dst[i] = y[i];
        }
        return;
    }
    ggml_expert_handoff_header * hdr = (ggml_expert_handoff_header *) mb;
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint32_t want = eh_dev_load(&hdr->seq_send);
        const uint64_t t0 = eh_dev_ticks();
        int good = 1;
        while (eh_dev_load(&hdr->seq_done) != want) {
            if (eh_dev_ticks() - t0 > timeout_ticks) {
                good = 0;
                break;
            }
            eh_dev_pause();
        }
        if (!good && blockIdx.x == 0) {
            eh_dev_store(&hdr->n_timeouts, eh_dev_load(&hdr->n_timeouts) + 1);
        }
        ok = good;
    }
    __syncthreads();

    const volatile float * my = (const volatile float *) (mb + off_y);
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        const int t = i / n_embd;
        const int c = i - t*n_embd;
        dst[i] = y[i] + (ok ? my[t*max_embd + c] : 0.0f);
    }
}

// device ticks per millisecond of eh_dev_ticks
static uint64_t eh_ticks_per_ms(const int device) {
#if defined(GGML_USE_HIP)
    static std::mutex mtx;
    static uint64_t rate[GGML_CUDA_MAX_DEVICES] = {};
    std::lock_guard<std::mutex> lk(mtx);
    if (rate[device] == 0) {
        int khz = 0;
        if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device) != hipSuccess || khz <= 0) {
            (void) hipGetLastError();
            khz = 100000; // RDNA and CDNA run the wall clock at 100 MHz
        }
        rate[device] = (uint64_t) khz;
    }
    return rate[device];
#elif defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    return 2000000;
#else
    GGML_UNUSED(device);
    return 1000000; // globaltimer counts nanoseconds
#endif
}

static void * eh_param_ptr(const ggml_tensor * dst, int lo_i, int hi_i) {
    const uint64_t lo = (uint32_t) ggml_get_op_params_i32(dst, lo_i);
    const uint64_t hi = (uint32_t) ggml_get_op_params_i32(dst, hi_i);
    return (void *) (uintptr_t) (lo | (hi << 32));
}

void ggml_cuda_op_expert_handoff(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    uint8_t * mb     = (uint8_t *) eh_param_ptr(dst, GGML_EXPERT_HANDOFF_P_PTR_LO,    GGML_EXPERT_HANDOFF_P_PTR_HI);
    int32_t * flags  = (int32_t *) eh_param_ptr(dst, GGML_EXPERT_HANDOFF_P_FLAGS_LO,  GGML_EXPERT_HANDOFF_P_FLAGS_HI);
    int32_t * counts = (int32_t *) eh_param_ptr(dst, GGML_EXPERT_HANDOFF_P_COUNTS_LO, GGML_EXPERT_HANDOFF_P_COUNTS_HI);
    const int layer  = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_LAYER);

    ggml_expert_handoff_dims d;
    d.max_tokens = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MAX_TOKENS);
    d.max_embd   = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MAX_EMBD);
    d.max_used   = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MAX_USED);
    const eh_offsets off = eh_offsets_of(d);

    if (ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MODE) == GGML_EXPERT_HANDOFF_SEND) {
        const ggml_tensor * x     = dst->src[0];
        const ggml_tensor * ids   = dst->src[1];
        const ggml_tensor * map   = dst->src[2];
        const ggml_tensor * w     = dst->src[3];

        const int n_embd   = (int) x->ne[0];
        const int n_tokens = (int) ggml_nrows(x);
        const int n_used   = (int) ids->ne[0];
        GGML_ASSERT(n_tokens <= d.max_tokens && n_embd <= d.max_embd && n_used <= d.max_used);

        k_expert_handoff_send<<<1, 1024, 0, stream>>>(
            (const float *) x->data, (const int32_t *) ids->data, (const int32_t *) map->data, (int) ggml_nelements(map), (const float *) w->data,
            (int32_t *) dst->data, mb, off, n_embd, n_tokens, n_used, layer, d.max_embd, d.max_used,
            ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MISS_SLOT), flags, counts,
            ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_COUNTS_STRIDE));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const ggml_tensor * y = dst->src[0];
    const int n_embd = (int) y->ne[0];
    const int n      = (int) ggml_nelements(y);
    const uint64_t timeout = (uint64_t) std::max(1, ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_TIMEOUT_MS))*eh_ticks_per_ms(ctx.device);

    const int threads = 256;
    const int blocks  = std::max(1, std::min((n + threads - 1)/threads, 32));
    k_expert_handoff_join<<<blocks, threads, 0, stream>>>(
        (const float *) y->data, (float *) dst->data, mb, off.y, n_embd, n, d.max_embd, timeout, flags, layer);
    CUDA_CHECK(cudaGetLastError());
}

void * ggml_cuda_host_mapped_alloc(size_t size) {
    void * ptr = nullptr;
#if defined(GGML_USE_HIP)
    cudaError_t err = hipHostMalloc(&ptr, size, hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent);
#else
    cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);
#endif
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: failed to allocate %.2f KiB of mapped host memory: %s\n", __func__, size/1024.0, cudaGetErrorString(err));
        return nullptr;
    }
    // every device reads and writes it at the host's address (the ops take one pointer, whichever GPU runs them)
    int cur = 0;
    CUDA_CHECK(cudaGetDevice(&cur));
    for (int id = 0; id < ggml_cuda_info().device_count; ++id) {
        ggml_cuda_set_device(id);
        void * dev = nullptr;
        err = cudaHostGetDevicePointer(&dev, ptr, 0);
        if (err != cudaSuccess || dev != ptr) {
            (void) cudaGetLastError();
            GGML_LOG_ERROR("%s: the mapped host memory has another address on device %d (%p vs %p)\n", __func__, id, dev, ptr);
            ggml_cuda_set_device(cur);
            CUDA_CHECK(cudaFreeHost(ptr));
            return nullptr;
        }
    }
    ggml_cuda_set_device(cur);
    memset(ptr, 0, size);
    return ptr;
}

void ggml_cuda_host_mapped_free(void * ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}
