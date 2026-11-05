#include "hc-up-pre.cuh"

#if defined(GGML_USE_HIP)

#define HCUP_DT      16    // embedding columns per block (x hc stream rows)
#define HCUP_TT      64    // tokens per block
#define HCUP_HC      4
#define HCUP_THREADS 256
#define HCUP_KMAX    384

// Q8_1 (D4) quantization of lo with the expressions of quantize_mmq_q8_1, one warp per 32 values.
// Written to a scratch buffer first: lo may share memory with the fused kernel's output (the allocator
// can reuse lo's buffer once the gate GEMM that reads it would have run).
static __global__ void hc_up_pre_quantize(const float * __restrict__ lo, const int64_t s_lo, int8_t * __restrict__ q,
        float * __restrict__ qd, const int n_tokens, const int K) {
    const int nkb = K / QK8_0;
    const int job = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (job >= n_tokens * nkb) {
        return;
    }
    const int t = job / nkb, kb = job % nkb;
    const float x = lo[(int64_t) t * s_lo + kb * QK8_0 + lane];
    float amax = fabsf(x);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
    }
    const float d_inv = 127.0f / amax;
    char qv = roundf(x * d_inv);
    q[(int64_t) t * K + kb * QK8_0 + lane] = qv;
    if (lane == 0) {
        qd[(int64_t) t * nkb + kb] = 1.0f / d_inv;
    }
}

// One block: HCUP_DT embedding columns x HCUP_HC streams of W rows and HCUP_TT tokens.
// Thread (dl, tg): column d0+dl, tokens t0+tg*4 .. +3, all HCUP_HC streams.
static __global__ void __launch_bounds__(HCUP_THREADS, 2) hc_up_pre_q8_kernel(
        const char * __restrict__ w, const int64_t w_row_bytes,
        const int8_t * __restrict__ loq, const float * __restrict__ lod,
        const float * __restrict__ xn, const int64_t sx1, const int64_t sx2,
        float * __restrict__ dst, const int64_t sd1,
        const int n_embd, const int n_tokens, const int K, const float scale) {
    extern __shared__ __align__(16) int hcup_smem[];
    const int nkb = K / QK8_0;
    int8_t * wq = (int8_t *) hcup_smem;                                   // [HC*DT][K]
    float  * wd = (float *) (wq + HCUP_HC * HCUP_DT * K);                 // [HC*DT][nkb]

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int d0 = blockIdx.x * HCUP_DT;
    const int t0 = blockIdx.y * HCUP_TT;

    // weights: rows c*n_embd + d0 + dl, stored word-major ([kb*8 + word][row]) so that the HCUP_DT threads
    // reading one word of consecutive rows hit consecutive LDS banks
    int * wq32 = (int *) wq;
    constexpr int NR = HCUP_HC * HCUP_DT;
    for (int idx = tid; idx < NR * nkb; idx += HCUP_THREADS) {
        const int r = idx % NR, kb = idx / NR;
        const int c = r / HCUP_DT, dl = r % HCUP_DT;
        const block_q8_0 * b = (const block_q8_0 *) (w + (int64_t) (c * n_embd + d0 + dl) * w_row_bytes) + kb;
        wd[r * nkb + kb] = __half2float(b->d);
        const uint16_t * src = (const uint16_t *) b->qs;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            wq32[(kb * 8 + i) * NR + r] = (int) src[2*i] | ((int) src[2*i + 1] << 16);
        }
    }
    __syncthreads();

    const int dl = tid % HCUP_DT;
    const int tg = tid / HCUP_DT;   // 0 .. TT/4 - 1
    float acc[HCUP_HC][4] = {{0.0f}};
    for (int kb = 0; kb < nkb; ++kb) {
        int wv[HCUP_HC][8];
#pragma unroll
        for (int c = 0; c < HCUP_HC; ++c) {
#pragma unroll
            for (int i = 0; i < 8; ++i) wv[c][i] = wq32[(kb * 8 + i) * NR + c * HCUP_DT + dl];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int tl = tg * 4 + j;
            // activations straight from the pre-quantized scratch (L1/L2 hits: 16 threads share each row);
            // past-the-end tokens read row 0 and are never stored
            const int tq = min(t0 + tl, n_tokens - 1);
            const int4 * a = (const int4 *) (loq + (int64_t) tq * K + kb * QK8_0);
            const int4 a0 = a[0], a1 = a[1];
            const int av[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
            const float dB = lod[(int64_t) tq * nkb + kb];
#pragma unroll
            for (int c = 0; c < HCUP_HC; ++c) {
                int isum = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) isum = ggml_cuda_dp4a(wv[c][i], av[i], isum);
                const float dA = wd[(c * HCUP_DT + dl) * nkb + kb];
                acc[c][j] += isum*dA*dB;
            }
        }
    }

    const int d = d0 + dl;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = t0 + tg * 4 + j;
        if (t >= n_tokens || d >= n_embd) {
            continue;
        }
        float sum = 0.0f;
#pragma unroll
        for (int ih = 0; ih < HCUP_HC; ++ih) {
            const float xv = xn[d + ih*sx1 + t*sx2];
            float wv2;
            wv2 = 1.0f / (1.0f + expf(-acc[ih][j]));
            sum += xv * wv2;
        }
        dst[d + t*sd1] = scale * sum;
    }
}

bool ggml_cuda_hc_up_pre_supported(const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * gate,
        const ggml_tensor * pre) {
    const ggml_tensor * x = pre->src[0];
    if (w->type != GGML_TYPE_Q8_0 || lo->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || pre->type != GGML_TYPE_F32) return false;
    if (ggml_get_op_params_i32(pre, 1) == 0) return false;   // gated form only
    const int64_t K = w->ne[0], n_embd = x->ne[0], hc = x->ne[1], T = x->ne[2];
    if (hc != HCUP_HC || K % QK8_0 != 0 || K > HCUP_KMAX || n_embd % HCUP_DT != 0) return false;
    if (w->ne[1] != hc * n_embd || lo->ne[0] != K || ggml_nrows(lo) != T || ggml_nelements(gate) != hc * n_embd * T) return false;
    if (!ggml_is_contiguous(w) || lo->nb[0] != sizeof(float) || x->nb[0] != sizeof(float) || pre->nb[0] != sizeof(float)) return false;
    if (lo->ne[2] * lo->ne[3] != 1 && lo->ne[1] != T) return false;
    if (pre->ne[0] != n_embd || pre->ne[1] != T) return false;
    // blocks write the output while other blocks still read W and xn
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    if (overlaps(pre, w) || overlaps(pre, x)) return false;
    return true;
}

void ggml_cuda_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo,
        const ggml_tensor * pre) {
    const ggml_tensor * x = pre->src[0];
    const int K = (int) w->ne[0], n_embd = (int) x->ne[0], T = (int) x->ne[2];
    const int nkb = K / QK8_0;
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int8_t> loq(ctx.pool(), (size_t) T * K);
    ggml_cuda_pool_alloc<float>  lod(ctx.pool(), (size_t) T * nkb);
    const int jobs = T * nkb;
    hc_up_pre_quantize<<<(jobs + 7) / 8, 256, 0, stream>>>((const float *) lo->data, (int64_t) (lo->nb[1] / sizeof(float)),
        loq.get(), lod.get(), T, K);
    const size_t smem = (size_t) HCUP_HC * HCUP_DT * K + (size_t) HCUP_HC * HCUP_DT * nkb * sizeof(float);
    const dim3 grid(n_embd / HCUP_DT, (T + HCUP_TT - 1) / HCUP_TT, 1);
    const float scale = ggml_get_op_params_f32(pre, 0);
    hc_up_pre_q8_kernel<<<grid, HCUP_THREADS, smem, stream>>>(
        (const char *) w->data, (int64_t) w->nb[1],
        loq.get(), lod.get(),
        (const float *) x->data, (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (x->nb[2] / sizeof(float)),
        (float *) pre->data, (int64_t) (pre->nb[1] / sizeof(float)),
        n_embd, T, K, scale);
    CUDA_CHECK(cudaGetLastError());
}

#else // !defined(GGML_USE_HIP)

bool ggml_cuda_hc_up_pre_supported(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) { return false; }
void ggml_cuda_hc_up_pre(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) {
    GGML_ABORT("hc_up_pre is HIP-only");
}

#endif // defined(GGML_USE_HIP)
