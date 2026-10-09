#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}


// rounds like the unfused MUL and ADD kernels (no contraction into fma)
static __device__ __forceinline__ float dsv4_mul_rn(const float a, const float b) {
#if defined(GGML_USE_HIP)
    float r;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
#else
    return __fmul_rn(a, b);
#endif
}

static __device__ __forceinline__ float dsv4_add_rn(const float a, const float b) {
#if defined(GGML_USE_HIP)
    float r;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
#else
    return __fadd_rn(a, b);
#endif
}

// mHC post gate and Sinkhorn comb of the same tokens in one kernel: the gate is
// gscale * sigmoid(x * s + b) + gbias as the unfused MUL, ADD, SIGMOID and SCALE round it, the comb as dsv4_hc_comb_f32
static __global__ void dsv4_hc_post_gate_comb_f32(
        const float * gx, const int64_t sgx1, const float * gs, const float * gb, float * gdst, const float gscale, const float gbias,
        const float * mixes, const float * scale, const float * base, float * dst, const int64_t n_tokens,
        const int64_t sm0, const int64_t sm1, const int64_t ss0, const int64_t sb0,
        const int64_t sd0, const int64_t sd1, const int64_t sd2, const float eps, const int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_tokens) {
        return;
    }

    for (int c = 0; c < DSV4_HC; ++c) {
        const float v = dsv4_add_rn(dsv4_mul_rn(gx[it*sgx1 + c], gs[0]), gb[c]);
        gdst[it*DSV4_HC + c] = dsv4_add_rn(dsv4_mul_rn(gscale, 1.0f / (1.0f + expf(-v))), gbias);
    }

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }
        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }
        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }
    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idst + DSV4_HC*isrc];
        }
    }
}

bool ggml_cuda_dsv4_hc_post_gate_comb(ggml_backend_cuda_context & ctx, const ggml_tensor * gx, const ggml_tensor * gs,
        const ggml_tensor * gb, ggml_tensor * gdst, const float gscale, const float gbias, ggml_tensor * comb) {
    const ggml_tensor * mixes = comb->src[0];
    const ggml_tensor * scale = comb->src[1];
    const ggml_tensor * base  = comb->src[2];
    const int64_t n_tokens = mixes->ne[1];
    if (comb->op != GGML_OP_DSV4_HC_COMB || gx->ne[0] != DSV4_HC || gx->ne[1] != n_tokens || gx->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(gdst) || ggml_nelements(gdst) != DSV4_HC * n_tokens || mixes->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || base->type != GGML_TYPE_F32 || comb->type != GGML_TYPE_F32) {
        return false;
    }
    const float   eps    = ggml_get_op_params_f32(comb, 0);
    const int32_t n_iter = ggml_get_op_params_i32(comb, 1);
    const int block_size = 64;
    dsv4_hc_post_gate_comb_f32<<<(unsigned) ((n_tokens + block_size - 1) / block_size), block_size, 0, ctx.stream()>>>(
        (const float *) gx->data, gx->nb[1] / sizeof(float), (const float *) gs->data, (const float *) gb->data, (float *) gdst->data,
        gscale, gbias, (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) comb->data,
        n_tokens, mixes->nb[0] / sizeof(float), mixes->nb[1] / sizeof(float), scale->nb[0] / sizeof(float), base->nb[0] / sizeof(float),
        comb->nb[0] / sizeof(float), comb->nb[1] / sizeof(float), comb->nb[2] / sizeof(float), eps, n_iter);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// mHC pre: the stream weights come from the pre gate chain (gscale * sigmoid(gx * gs + gb) + gbias, rounded as the
// unfused MUL, ADD, SIGMOID and SCALE), computed here instead of being read from that chain's output
static __global__ void dsv4_hc_pre_gate_f32(
        const float * x, const float * gx, const int64_t sgx1, const float * gs, const float * gb, const float gscale, const float gbias,
        float * dst, const int64_t n_embd, const int64_t hc, const int64_t n_tokens, const int64_t sx0, const int64_t sx1,
        const int64_t sx2, const int64_t sd0, const int64_t sd1, const float scale) {
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;
    if (ir >= nr) {
        return;
    }
    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        const float v  = dsv4_add_rn(dsv4_mul_rn(gx[it*sgx1 + ih], gs[0]), gb[ih]);
        const float wv = dsv4_add_rn(dsv4_mul_rn(gscale, 1.0f / (1.0f + expf(-v))), gbias);
        sum += xv * wv;
    }
    dst[i0*sd0 + it*sd1] = scale * sum;
}

bool ggml_cuda_dsv4_hc_pre_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * gx, const ggml_tensor * gs,
        const ggml_tensor * gb, const ggml_tensor * gate, const float gscale, const float gbias, ggml_tensor * pre) {
    const ggml_tensor * x = pre->src[0];
    if (pre->op != GGML_OP_DSV4_HC_PRE || pre->src[1] != gate || ggml_get_op_params_i32(pre, 1) != 0 ||
            x->type != GGML_TYPE_F32 || pre->type != GGML_TYPE_F32 || gx->nb[0] != sizeof(float) ||
            gx->ne[0] != x->ne[1] || gx->ne[1] != x->ne[2] || gate->ne[0] != x->ne[1] || !ggml_is_contiguous(gate)) {
        return false;
    }
    const int64_t n_embd = x->ne[0], hc = x->ne[1], n_tokens = x->ne[2];
    const float   scale  = ggml_get_op_params_f32(pre, 0);
    const int64_t nr     = n_embd * n_tokens;
    dsv4_hc_pre_gate_f32<<<(unsigned) ((nr + 255) / 256), 256, 0, ctx.stream()>>>((const float *) x->data, (const float *) gx->data,
        gx->nb[1] / sizeof(float), (const float *) gs->data, (const float *) gb->data, gscale, gbias, (float *) pre->data,
        n_embd, hc, n_tokens, x->nb[0] / sizeof(float), x->nb[1] / sizeof(float), x->nb[2] / sizeof(float),
        pre->nb[0] / sizeof(float), pre->nb[1] / sizeof(float), scale);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

template <bool has_comb>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

// the same sums, one thread per (embedding column, token) for all DSV4_HC outputs: each residual value is read once
// and there is no 64-bit index division (prefill at ubatch 4096: one call moves about 0.5 GB)
template <bool has_comb>
static __global__ void dsv4_hc_post4_f32(
        const float * x, const float * residual, const float * post, const float * comb, float * dst,
        const int n_embd, const int64_t sx0, const int64_t sx1, const int64_t sr0, const int64_t sr1, const int64_t sr2,
        const int64_t sp0, const int64_t sp1, const int64_t sc0, const int64_t sc1, const int64_t sc2,
        const int64_t sd0, const int64_t sd1, const int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t it = blockIdx.y;
    if (i0 >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float xv = x[i0*sx0 + it*sx1];
    float r[DSV4_HC];
#pragma unroll
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        r[isrc] = residual[i0*sr0 + isrc*sr1 + it*sr2];
    }
#pragma unroll
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = xv * post[idst*sp0 + it*sp1];
        if constexpr (has_comb) {
#pragma unroll
            for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
                sum += r[isrc] * comb[idst*sc0 + isrc*sc1 + it*sc2];
            }
        } else {
            sum += r[idst];
        }
        dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
    }
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    if (hc == DSV4_HC && n_embd <= INT_MAX && n_tokens <= 65535) {
        const dim3 grid4((n_embd + block_size - 1) / block_size, n_tokens, 1);
        const ggml_cuda_kernel_launch_params params4(grid4, dim3(block_size, 1, 1), 0, ctx.stream());
        auto kernel4 = comb ? dsv4_hc_post4_f32<true> : dsv4_hc_post4_f32<false>;
        ggml_cuda_kernel_launch(kernel4, params4,
                (const float *) x->data, (const float *) residual->data,
                (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
                (int) n_embd,
                nbx0 / sizeof(float), nbx1 / sizeof(float),
                nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
                nbp0 / sizeof(float), nbp1 / sizeof(float),
                nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
                nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
        return;
    }
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}
