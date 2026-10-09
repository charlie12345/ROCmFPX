#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "qsa.cuh"
#include "fattn-glm512.cuh"

#if !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
// (HIP: wave32 devices only, the ballot words are 32 lanes)
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
#if defined(GGML_USE_HIP)
            selected_warp[item] = uint32_t(__ballot(selected));
#else
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
#endif // defined(GGML_USE_HIP)
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}

// Long masks (decode deep in a DSA context): the single-block scan above walks every column in sequence. Here many
// blocks each take a chunk of FATTN_COMPACT_CHUNK columns: the first pass counts the visible columns per chunk, the
// second writes each chunk's indices after the counts of the chunks before it, so the list keeps ascending order.
#define FATTN_COMPACT_CHUNK 8192

template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_chunk_count(
        const half * mask_ptr, int32_t * chunk_counts, const int ne30, const int n_queries, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();
    const int group = blockIdx.x, chunk = blockIdx.y, sequence = blockIdx.z;
    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);
    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    const int c0 = chunk*FATTN_COMPACT_CHUNK, c1 = min(ne30, c0 + FATTN_COMPACT_CHUNK);

    int n = 0;
    for (int i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
        bool selected = false;
#pragma unroll
        for (int q = 0; q < ncols1; ++q) {
            selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
        }
        n += selected;
    }
    __shared__ int part[256/WARP_SIZE];
    n = warp_reduce_sum<WARP_SIZE>(n);
    if (threadIdx.x % WARP_SIZE == 0) {
        part[threadIdx.x / WARP_SIZE] = n;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < 256/WARP_SIZE; ++w) {
            total += part[w];
        }
        chunk_counts[(int64_t(sequence)*gridDim.x + group)*gridDim.y + chunk] = total;
    }
}

template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_chunk_write(
        const half * mask_ptr, const int32_t * chunk_counts, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30,
        const int n_queries, const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();   // reads the per-chunk counts
    constexpr int values_per_lane = 8;
    const int tid = threadIdx.x, warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const int group = blockIdx.x, chunk = blockIdx.y, sequence = blockIdx.z;
    const int n_chunks = gridDim.y;
    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);
    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    const int64_t list = int64_t(sequence)*gridDim.x + group;
    int32_t * indices = indices_ptr + list*n_kv_max;
    const int c0 = chunk*FATTN_COMPACT_CHUNK, c1 = min(ne30, c0 + FATTN_COMPACT_CHUNK);

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;
    if (tid == 0) {
        int off = 0;
        for (int c = 0; c < chunk; ++c) {
            off += chunk_counts[list*n_chunks + c];
        }
        row_count = off;
    }
    __syncthreads();

    for (int i0 = c0; i0 < c1; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            bool selected = false;
            if (i < c1) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
#if defined(GGML_USE_HIP)
            selected_warp[item] = uint32_t(__ballot(selected));
#else
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
#endif // defined(GGML_USE_HIP)
            warp_count += __popc(selected_warp[item]);
        }
        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();
        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();
        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();
        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    // the last chunk knows the total: pad the list and store its length
    if (chunk == n_chunks - 1) {
        const int count = min(row_count, n_kv_max);
        for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
            indices[i] = -1;
        }
        if (tid == 0) {
            counts_ptr[list] = count;
        }
    }
    __syncthreads();
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_MUSA)

// Sparse decode reads only the listed rows: convert just those q8_0 K rows to f16, in list order, with the matching
// mask columns of every query, and point the lists at the compact copies.
static __global__ void flash_attn_sparse_gather_q8_0(
        const char * __restrict__ K, const int64_t nbk1, const half * __restrict__ mask, const int64_t s31, const int n_queries,
        int32_t * __restrict__ indices, half * __restrict__ K_c, half * __restrict__ mask_c, const int n_kv_max, const int n_cols_c) {
    const int list = blockIdx.x;
    const int slot = blockIdx.y * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (slot >= n_kv_max) {
        return;
    }
    const int pos = list*n_kv_max + slot;
    const int row = indices[pos];
    if (row < 0) {
        return;
    }
    // 512 values = 16 q8_0 blocks; lane converts half a block
    const block_q8_0 * kb = (const block_q8_0 *) (K + row*nbk1);
    const int b = lane / 2, o = (lane % 2) * 16;
    const float d = __half2float(kb[b].d);
    half * kc = K_c + (int64_t) pos*512 + b*32 + o;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        kc[k] = __float2half(d * (float) kb[b].qs[o + k]);
    }
    if (lane < n_queries) {
        mask_c[(int64_t) lane*n_cols_c + pos] = mask[lane*s31 + row];
    }
    __syncwarp();
    if (lane == 0) {
        indices[pos] = pos;
    }
}

void ggml_cuda_flash_attn_ext_sparse_gather(
        const ggml_tensor * K, const ggml_tensor * mask, const int n_queries, int32_t * indices, const int n_lists, const int n_kv_max,
        half * K_c, half * mask_c, cudaStream_t stream) {
    GGML_ASSERT(K->type == GGML_TYPE_Q8_0 && K->ne[0] == 512 && n_queries <= 32);
    const dim3 grid(n_lists, (n_kv_max + 7) / 8, 1);
    flash_attn_sparse_gather_q8_0<<<grid, 256, 0, stream>>>((const char *) K->data, K->nb[1], (const half *) mask->data,
        mask->nb[1] / sizeof(half), n_queries, indices, K_c, mask_c, n_kv_max, n_lists*n_kv_max);
    CUDA_CHECK(cudaGetLastError());
}

int ggml_cuda_flash_attn_ext_compact_mask_scratch(const ggml_tensor * mask) {
    const int n_chunks = (int) ((mask->ne[0] + FATTN_COMPACT_CHUNK - 1) / FATTN_COMPACT_CHUNK);
    return n_chunks > 1 ? n_chunks : 0;
}

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream,
        int32_t * chunk_counts) {
#if defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream, chunk_counts);
    GGML_ABORT("sparse flash attention is not supported on MUSA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // the last group of queries is partial only if ncols1 does not divide n_queries
    GGML_ASSERT(ncols1 == 1 || ncols1 == 2 || ncols1 == 4 || ncols1 == 8);
    const bool oob = n_queries % ncols1 != 0;
    const int n_chunks = ggml_cuda_flash_attn_ext_compact_mask_scratch(mask);
    if (chunk_counts != nullptr && n_chunks > 1) {
        const dim3 grid((n_queries + ncols1 - 1)/ncols1, n_chunks, mask->ne[3]);
        const ggml_cuda_kernel_launch_params chunk_params(grid, block_dim, 0, stream);
#define FATTN_COMPACT_LAUNCH(NC, OOB) \
        ggml_cuda_kernel_launch(flash_attn_mask_chunk_count<NC, OOB>, chunk_params, (const half *) mask->data, chunk_counts, int(mask->ne[0]), n_queries, s31, s33); \
        ggml_cuda_kernel_launch(flash_attn_mask_chunk_write<NC, OOB>, chunk_params, (const half *) mask->data, (const int32_t *) chunk_counts, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33)
        switch (ncols1) {
            case 1:  FATTN_COMPACT_LAUNCH(1, false); break;
            case 2:  if (oob) { FATTN_COMPACT_LAUNCH(2, true); } else { FATTN_COMPACT_LAUNCH(2, false); } break;
            case 4:  if (oob) { FATTN_COMPACT_LAUNCH(4, true); } else { FATTN_COMPACT_LAUNCH(4, false); } break;
            default: if (oob) { FATTN_COMPACT_LAUNCH(8, true); } else { FATTN_COMPACT_LAUNCH(8, false); } break;
        }
#undef FATTN_COMPACT_LAUNCH
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const auto kernel = ncols1 == 1 ? flash_attn_mask_to_sparse_indices<1, false> :
                        ncols1 == 2 ? (oob ? flash_attn_mask_to_sparse_indices<2, true> : flash_attn_mask_to_sparse_indices<2, false>) :
                        ncols1 == 4 ? (oob ? flash_attn_mask_to_sparse_indices<4, true> : flash_attn_mask_to_sparse_indices<4, false>) :
                                      (oob ? flash_attn_mask_to_sparse_indices<8, true> : flash_attn_mask_to_sparse_indices<8, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2*n_gather);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

#define FATTN_VEC_CASE_CUSTOM(D, type_K_enum, type_V_enum) \
    if (head_size == (D) && type_K == (type_K_enum) && type_V == (type_V_enum)) { \
        return ggml_cuda_flash_attn_ext_vec_case<D, type_K_enum, type_V_enum>; \
    }

#define FATTN_VEC_CASES_CUSTOM_ALL_D(type_K_enum, type_V_enum) \
    FATTN_VEC_CASE_CUSTOM( 64, type_K_enum, type_V_enum) \
    FATTN_VEC_CASE_CUSTOM(128, type_K_enum, type_V_enum) \
    FATTN_VEC_CASE_CUSTOM(256, type_K_enum, type_V_enum)

#define FATTN_VEC_CASES_CUSTOM_TURBO(type_K_enum, type_V_enum) \
    FATTN_VEC_CASE_CUSTOM(128, type_K_enum, type_V_enum) \
    FATTN_VEC_CASE_CUSTOM(256, type_K_enum, type_V_enum)

#ifdef GGML_USE_HIP
    FATTN_VEC_CASES_CUSTOM_ALL_D(GGML_TYPE_Q4_0_ROCMFP4,      GGML_TYPE_Q4_0_ROCMFP4)
    FATTN_VEC_CASES_CUSTOM_ALL_D(GGML_TYPE_Q4_0_ROCMFP4_FAST, GGML_TYPE_Q4_0_ROCMFP4_FAST)
    FATTN_VEC_CASES_CUSTOM_ALL_D(GGML_TYPE_Q3_0_ROCMFPX,      GGML_TYPE_Q3_0_ROCMFPX)
    FATTN_VEC_CASES_CUSTOM_ALL_D(GGML_TYPE_Q6_0_ROCMFPX,      GGML_TYPE_Q6_0_ROCMFPX)
    FATTN_VEC_CASES_CUSTOM_ALL_D(GGML_TYPE_Q8_0_ROCMFPX,      GGML_TYPE_Q8_0_ROCMFPX)

    FATTN_VEC_CASES_CUSTOM_TURBO(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_CUSTOM_TURBO(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_CUSTOM_TURBO(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_CUSTOM_TURBO(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_CUSTOM_TURBO(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_CUSTOM_TURBO(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_CUSTOM_TURBO(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_CUSTOM_TURBO(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
#endif

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
#ifdef GGML_USE_HIP
static bool ggml_type_is_turbo(ggml_type type) {
    return type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
}

static bool ggml_cuda_fattn_turbo_vec_pair_supported(ggml_type type_k, ggml_type type_v) {
    const bool k_supported = type_k == GGML_TYPE_Q8_0 || ggml_type_is_turbo(type_k);
    const bool v_supported = type_v == GGML_TYPE_Q8_0 || ggml_type_is_turbo(type_v);
    return (ggml_type_is_turbo(type_k) || ggml_type_is_turbo(type_v)) && k_supported && v_supported;
}

// TurboQuant SET_ROWS encodes independent normalized 128-point FWHT chunks.
// This kernel is self-inverse: it rotates Q into the packed cache domain and,
// after attention, rotates the accumulated V result back to model space.
static __global__ void k_fattn_turbo_fwht_128(
        const char * __restrict__ src,
        float      * __restrict__ dst,
        int64_t ne0, int64_t ne1, int64_t ne2,
        int64_t nb1, int64_t nb2, int64_t nb3) {
    __shared__ float values[128];

    const int tid = threadIdx.x;
    const int64_t chunks_per_row = ne0 / 128;
    const int64_t chunk = blockIdx.x % chunks_per_row;
    int64_t row = blockIdx.x / chunks_per_row;
    const int64_t i1 = row % ne1;
    row /= ne1;
    const int64_t i2 = row % ne2;
    const int64_t i3 = row / ne2;

    const float * src_row = (const float *) (src + i1*nb1 + i2*nb2 + i3*nb3);
    values[tid] = src_row[chunk*128 + tid];
    __syncthreads();

#pragma unroll
    for (int h = 1; h < 128; h <<= 1) {
        if (tid < 64) {
            const int group = tid / h;
            const int pos   = tid % h;
            const int i0    = group*2*h + pos;
            const float a   = values[i0];
            const float b   = values[i0 + h];
            values[i0]      = a + b;
            values[i0 + h]  = a - b;
        }
        __syncthreads();
    }

    dst[(int64_t) blockIdx.x*128 + tid] = values[tid] * 0.08838834764831844f;
}

static void ggml_cuda_fattn_turbo_fwht_128(
        const ggml_tensor * src, float * dst, cudaStream_t stream) {
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(src->ne[0] == 128 || src->ne[0] == 256);
    GGML_ASSERT(src->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_nelements(src) % 128 == 0);

    const int64_t nblocks = ggml_nelements(src) / 128;
    k_fattn_turbo_fwht_128<<<nblocks, 128, 0, stream>>>(
        (const char *) src->data, dst,
        src->ne[0], src->ne[1], src->ne[2],
        src->nb[1], src->nb[2], src->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

static ggml_tensor ggml_cuda_fattn_turbo_rotate_q(
        const ggml_tensor * src, ggml_cuda_pool_alloc<float> & pool_buf, cudaStream_t stream) {
    pool_buf.alloc(ggml_nelements(src));
    ggml_cuda_fattn_turbo_fwht_128(src, pool_buf.ptr, stream);

    ggml_tensor tmp = *src;
    tmp.data  = pool_buf.ptr;
    tmp.nb[0] = sizeof(float);
    tmp.nb[1] = tmp.ne[0] * tmp.nb[0];
    tmp.nb[2] = tmp.ne[1] * tmp.nb[1];
    tmp.nb[3] = tmp.ne[2] * tmp.nb[2];
    tmp.view_src  = nullptr;
    tmp.view_offs = 0;
    return tmp;
}

static bool ggml_cuda_fattn_turbo_vec_fused_eligible(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    return ggml_cuda_fattn_turbo_vec_pair_supported(K->type, V->type) &&
           (K->ne[0] == 128 || K->ne[0] == 256) &&
           V->ne[0] == K->ne[0] && Q->ne[0] == K->ne[0] &&
           K->ne[1] % FATTN_KQ_STRIDE == 0 &&
           Q->nb[0] == ggml_element_size(Q) &&
           K->nb[0] == ggml_element_size(K) &&
           V->nb[0] == ggml_element_size(V) &&
           ggml_is_contiguous(dst);
}

static bool ggml_cuda_fattn_turbo_batched_required(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool q8_turbo_mixed = (K->type == GGML_TYPE_Q8_0 && ggml_type_is_turbo(V->type)) ||
                                (V->type == GGML_TYPE_Q8_0 && ggml_type_is_turbo(K->type));

    return ggml_cuda_fattn_turbo_vec_fused_eligible(dst) && Q->ne[1] > 8 &&
           (K->ne[1] > 1024 || q8_turbo_mixed);
}
#endif // GGML_USE_HIP

enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    const bool is_rocmfp_family = K->type == GGML_TYPE_Q4_0_ROCMFP4 ||
                                  K->type == GGML_TYPE_Q4_0_ROCMFP4_FAST ||
                                  K->type == GGML_TYPE_Q3_0_ROCMFPX ||
                                  K->type == GGML_TYPE_Q6_0_ROCMFPX ||
                                  K->type == GGML_TYPE_Q8_0_ROCMFPX;

    if (is_rocmfp_family) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_NONE;
    }

#ifdef GGML_USE_HIP
    // Decode and speculative batches consume packed Turbo KV directly. Larger
    // prompt batches use bounded-slice TILE conversion in the dispatch path.
    if (ggml_cuda_fattn_turbo_vec_fused_eligible(dst) && can_use_vector_kernel && Q->ne[1] <= 8) {
        return BEST_FATTN_KERNEL_VEC;
    }

    // HIP graph capture cannot tolerate cudaMalloc/cudaFree in the TILE/MMA
    // quantized-KV f16 temp path. Route small decode/speculative batches through
    // VEC, which dequantizes inline and avoids capture-unsafe allocation calls.
    // Large prefill batches still fall through to the faster TILE/MMA path.
    if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) && can_use_vector_kernel && Q->ne[1] <= 8) {
        return BEST_FATTN_KERNEL_VEC;
    }
#endif // GGML_USE_HIP

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    if (GGML_CUDA_CC_IS_RDNA3_5(cc) && gqa_opt_applies && Q->ne[0] == 256 && V->ne[0] == 256 && Q->ne[1] * gqa_ratio_eff > 32) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT || dst->op == GGML_OP_FLASH_ATTN_EXT_PARTIAL);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    if (dst->op == GGML_OP_FLASH_ATTN_EXT_PARTIAL) {
        // only the 512-wide partial path runs the tile kernel on f16 copies of K/V
        const bool tile = Q->ne[0] == 512;
        return ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, tile, tile).end - (uintptr_t) dst->data;
    }

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    if (ggml_cuda_flash_attn_ext_qsa_decode_supported(ctx, dst)) {
        ggml_cuda_flash_attn_ext_qsa_decode(ctx, dst);
        return;
    }
    if (ggml_cuda_flash_attn_ext_qsa_supported(ctx, dst)) {
        ggml_cuda_flash_attn_ext_qsa(ctx, dst);
        return;
    }
    if (ggml_cuda_flash_attn_ext_glm512(ctx, dst)) {
        return;
    }
    // only the qsa kernel honours the selected-cell indices; the kernels below attend to every key, so
    // a maskless sparse op reaching them would read cells the mask exists to hide
    GGML_ASSERT((dst->src[3] || !dst->src[5]) && "sparse flash attention without a mask needs the qsa kernel");
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

bool ggml_cuda_flash_attn_ext_partial_supported(int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    GGML_UNUSED(device);
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (q->ne[0] == 512) {
        // MLA latents (V may be a view of K) through the tile kernel, under the GQA conditions of ordinary 512 attention
        return q->type == GGML_TYPE_F32 && k->type == v->type && (k->type == GGML_TYPE_Q4_0 || k->type == GGML_TYPE_Q8_0) &&
            k->ne[0] == 512 && v->ne[0] == 512 && k->ne[1] % FATTN_KQ_STRIDE == 0 && q->ne[2] % k->ne[2] == 0 &&
            (q->ne[2] / k->ne[2]) % 2 == 0 && mask && mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && max_bias == 0.0f &&
            !dst->src[4] && !dst->src[5] && ggml_get_op_params_i32(dst, 4) == 0;
    }
    return q->type == GGML_TYPE_F32 && k->type == v->type &&
        (k->type == GGML_TYPE_Q4_0 || k->type == GGML_TYPE_Q8_0) &&
        (q->ne[0] == 64 || q->ne[0] == 128 || q->ne[0] == 256) &&
        q->ne[0] == k->ne[0] && k->ne[0] == v->ne[0] && k->ne[1] % FATTN_KQ_STRIDE == 0 &&
        (!mask || (mask->type == GGML_TYPE_F16 && mask->ne[2] == 1)) && !dst->src[5] &&
        ggml_get_op_params_i32(dst, 4) == 0 && (!dst->src[4] || ggml_is_contiguous(dst->src[4]));
#endif
}

template <int D>
static void ggml_cuda_flash_attn_partial_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[1]->type == GGML_TYPE_Q4_0) {
        ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, true>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true>(ctx, dst);
    }
}

void ggml_cuda_flash_attn_ext_partial(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    GGML_ASSERT(ggml_cuda_flash_attn_ext_partial_supported(ctx.device, dst));
    switch (dst->src[0]->ne[0]) {
        case 64: ggml_cuda_flash_attn_partial_case<64>(ctx, dst); break;
        case 128: ggml_cuda_flash_attn_partial_case<128>(ctx, dst); break;
        case 256: ggml_cuda_flash_attn_partial_case<256>(ctx, dst); break;
        case 512:
            if (!ggml_cuda_flash_attn_ext_glm512(ctx, dst)) {
                ggml_cuda_flash_attn_ext_tile(ctx, dst);
            }
            break;
        default: GGML_ABORT("unsupported partial attention head size");
    }
}

static __global__ void attn_merge(const float * a, const float * b, float * dst, int width, bool normalize) {
    const int64_t row = blockIdx.x;
    const int64_t stride = width + 2;
    a += row*stride;
    b += row*stride;
    dst += row*(normalize ? width : stride);
    const float la = a[width+1], lb = b[width+1];
    const float m = fmaxf(la > 0 ? a[width] : -INFINITY, lb > 0 ? b[width] : -INFINITY);
    const float wa = la > 0 ? expf(a[width] - m) : 0;
    const float wb = lb > 0 ? expf(b[width] - m) : 0;
    const float l = wa*la + wb*lb;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        const float o = (wa > 0 ? wa*a[d] : 0) + (wb > 0 ? wb*b[d] : 0);
        dst[d] = normalize ? (l > 0 ? o/l : 0) : o;
    }
    if (!normalize && threadIdx.x == 0) { dst[width] = m; dst[width+1] = l; }
}

void ggml_cuda_attn_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    const auto * a = dst->src[0];
    const auto * b = dst->src[1];
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && ggml_is_contiguous(a) && ggml_is_contiguous(b));
    const dim3 blocks(ggml_nrows(a), 1, 1);
    const dim3 threads(128, 1, 1);
    const auto launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(attn_merge, launch_params, (const float *) a->data, (const float *) b->data,
        (float *) dst->data, int(a->ne[0] - 2), ggml_get_op_params_i32(dst, 0) != 0);
    CUDA_CHECK(cudaGetLastError());
}
