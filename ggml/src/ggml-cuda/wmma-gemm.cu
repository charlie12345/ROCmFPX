#include "wmma-gemm.cuh"
#include "convert.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// [TAG_WMMA_GEMM] The library is loaded at run time from its per-arch build (rocm_wmma_gemm_<arch>_accf32: f16 inputs,
// f32 accumulation and output), next to the executable or in ROCMFPX_WMMA_GEMM_DIR. C entry point, layouts 0 = row
// major, 1 = column major, stream passed by address.
typedef void (*wmma_gemm_f32_f16_t)(void * C, void * A, void * B, size_t M, size_t N, size_t K, size_t batch_count,
                                    int layout_c, int layout_a, int layout_b, void * stream);

static int wmma_gemm_env(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

// Every gfx11 chip (RDNA3, RDNA3.5): the library uses the gfx11 WMMA builtins, which all of them have and RDNA4 (gfx12,
// other operand vectors) does not. Configurations are tuned upstream for gfx1100 and gfx1151 only; wmma-gemm.cmake builds
// the desktop chips (gfx1101, gfx1102) with the gfx1100 ones and the other iGPUs (gfx1103, gfx1150, gfx1152) with the
// gfx1151 ones. Speed off gfx1100 is not measured; a chip whose library was not built stays on MMQ.
static const char * wmma_gemm_arch(const int cc) {
    switch (cc - GGML_CUDA_CC_OFFSET_AMD) {
        case 0x1100: return "gfx1100";
        case 0x1101: return "gfx1101";
        case 0x1102: return "gfx1102";
        case 0x1103: return "gfx1103";
        case 0x1150: return "gfx1150";
        case 0x1151: return "gfx1151";
        case 0x1152: return "gfx1152";
        default:     return nullptr;
    }
}

// The directory of the module this code is in (ggml-hip): the build puts the library next to it.
static std::string wmma_gemm_module_dir() {
#ifdef _WIN32
    HMODULE self = nullptr;
    char path[MAX_PATH] = {};
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR) &wmma_gemm_module_dir, &self) && GetModuleFileNameA(self, path, MAX_PATH)) {
        std::string p(path);
        const size_t slash = p.find_last_of("\\/");
        return slash == std::string::npos ? std::string() : p.substr(0, slash);
    }
#else
    Dl_info info;
    if (dladdr((void *) &wmma_gemm_module_dir, &info) && info.dli_fname) {
        std::string p(info.dli_fname);
        const size_t slash = p.find_last_of('/');
        return slash == std::string::npos ? std::string() : p.substr(0, slash);
    }
#endif
    return std::string();
}

static wmma_gemm_f32_f16_t wmma_gemm_load(const char * arch) {
    static std::mutex mutex;
    static std::map<std::string, wmma_gemm_f32_f16_t> loaded;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = loaded.find(arch);
    if (it != loaded.end()) {
        return it->second;
    }
    const char * env_dir = getenv("ROCMFPX_WMMA_GEMM_DIR");
    const std::string dir = env_dir ? std::string(env_dir) : wmma_gemm_module_dir();
    const std::string name = std::string("rocm_wmma_gemm_") + arch + "_accf32";
    wmma_gemm_f32_f16_t fn = nullptr;
#ifdef _WIN32
    const std::string path = (dir.empty() ? std::string() : dir + "\\") + name + ".dll";
    HMODULE h = LoadLibraryA(path.c_str());
    if (h) {
        fn = (wmma_gemm_f32_f16_t) GetProcAddress(h, "rocm_wmma_gemm_f32_f16");
    }
#else
    const std::string path = (dir.empty() ? std::string() : dir + "/") + "lib" + name + ".so";
    void * h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h) {
        fn = (wmma_gemm_f32_f16_t) dlsym(h, "rocm_wmma_gemm_f32_f16");
    }
#endif
    GGML_LOG_INFO("%s: %s: %s\n", __func__, path.c_str(),
        fn ? "loaded, prompt-sized quantized products use the WMMA GEMM" : "not available, MMQ as before");
    loaded[arch] = fn;
    return fn;
}

// [TAG_WMMA_GEMM_FP8_EMU] What an FP8 version of this path would cost in quality, measured on gfx1100 before writing one
// for RDNA4 (whose FP8 WMMA runs at twice its FP16 rate): ROCMFPX_WMMA_GEMM_FP8_EMU=1 rounds the unpacked weights to
// OCP E4M3 (3 mantissa bits, saturating at 448), 2 the activations (once, from f32), 3 both, each row with its own power-of-two scale
// (an output channel of the weights, a token of the activations), so the scale itself adds no error. The block scales
// of the quant are already folded into the f16 weights, as an FP8 unpack would fold them. The GEMM stays f16.
static __device__ __forceinline__ float fp8_e4m3_round(const float x) {
    const float a = fabsf(x);
    if (!(a > 0.0f)) {
        return 0.0f;
    }
    int e;
    frexpf(a, &e);                                     // a = m * 2^e, m in [0.5, 1)
    const float q = ldexpf(1.0f, max(e - 1, -6) - 3);  // spacing of 3 mantissa bits; the subnormals share 2^-6's
    return copysignf(fminf(rintf(a / q) * q, 448.0f), x);
}

// x rows rounded in place, or (activations) read as f32 from xf and rounded once, as an FP8 path would quantize them
template <int nthreads>
static __global__ void fp8_e4m3_emulate_rows(half * __restrict__ x, const float * __restrict__ xf, const int64_t ncols) {
    half * row = x + blockIdx.x*ncols;
    const float * rowf = xf ? xf + blockIdx.x*ncols : nullptr;
    float amax = 0.0f;
    for (int64_t i = threadIdx.x; i < ncols; i += nthreads) {
        amax = fmaxf(amax, fabsf(rowf ? rowf[i] : __half2float(row[i])));
    }
    __shared__ __align__(16) float smax[nthreads/WARP_SIZE];
    amax = warp_reduce_max(amax);
    if (threadIdx.x % WARP_SIZE == 0) {
        smax[threadIdx.x / WARP_SIZE] = amax;
    }
    __syncthreads();
    amax = 0.0f;
#pragma unroll
    for (int w = 0; w < nthreads/WARP_SIZE; ++w) {
        amax = fmaxf(amax, smax[w]);
    }
    if (!(amax > 0.0f)) {
        return;
    }
    int e;
    frexpf(448.0f / amax, &e);                         // the largest power of two s with amax*s <= 448
    const float s   = ldexpf(1.0f, e - 1);
    const float inv = ldexpf(1.0f, 1 - e);
    for (int64_t i = threadIdx.x; i < ncols; i += nthreads) {
        row[i] = __float2half(fp8_e4m3_round((rowf ? rowf[i] : __half2float(row[i])) * s) * inv);
    }
}

static void fp8_e4m3_emulate_rows_cuda(half * x, const float * xf, const int64_t nrows, const int64_t ncols, cudaStream_t stream) {
    constexpr int nthreads = 256;
    fp8_e4m3_emulate_rows<nthreads><<<(unsigned) nrows, nthreads, 0, stream>>>(x, xf, ncols);
}

// [TAG_FP8_GEMM] The prompt path for RDNA4, whose FP8 WMMA runs at twice its FP16 rate (RX 9070 XT: 389 against 195
// TFLOPS dense; int8 is 389 too, but MMQ applies a scale every 32 values, as an FP8 kernel would have to for the quant
// blocks). Weights are unpacked to f16 as for the WMMA GEMM, then to OCP E4M3 with one power-of-two scale per weight row,
// so the block scales are folded into the values; activations go to E4M3 with one per token. The GEMM is plain FP8
// with f32 accumulation and the two scales applied on the way out. Quality as [TAG_WMMA_GEMM_FP8_EMU] measures it.
//
// The kernel follows the design AMD describes for its RDNA4 FP8 GEMM (ROCm blog "FP8 GEMM Optimization on AMD RDNA 4",
// written here from the description): both operands K-contiguous (activation rows, weight rows as stored), 16-byte
// global loads into one LDS buffer whose rows are padded by 16 bytes, tiles taken in bands of 8 rows for L2 reuse, the
// next K step loaded into registers while the current one computes and committed after it. A lane's 16 bytes of a
// 32-wide K step feed two WMMAs (bytes 0-7, then 8-15), the K order the fork's RDNA4 int8 MMA uses: the two halves of
// the wave hold K 0-15 and 16-31, so the first WMMA sums K 0-7 and 16-23, the second 8-15 and 24-31.
//
// Written without an RDNA4 card. On RDNA3 the same kernel runs with each E4M3 value widened to f16 (exact) and the f16
// WMMA, to check everything but the gfx12 fragment layout: ROCMFPX_FP8_GEMM=1 there gives the numbers of
// ROCMFPX_WMMA_GEMM_FP8_EMU=3 to f32 rounding. Off by default (ROCMFPX_FP8_GEMM=1 turns it on) until it is measured on RDNA4.
static __device__ __forceinline__ uint8_t fp8_e4m3_encode(const float x) {
    // x pre-scaled to |x| <= 448; nearest, ties to even, as fp8_e4m3_round
    const uint32_t sign = x < 0.0f ? 0x80u : 0u;
    const float a = fabsf(x);
    if (!(a > 0.0f)) {
        return 0;
    }
    int e;
    frexpf(a, &e);
    int E = e - 1;
    if (E < -6) {
        // subnormal steps of 2^-9; a carry to 8 is the smallest normal, 0x08
        return (uint8_t) (sign | (uint32_t) rintf(a * 512.0f));
    }
    int m = (int) rintf(ldexpf(a, 3 - E));             // 8..16
    if (m == 16) {
        m = 8;
        ++E;
    }
    if (E > 8 || (E == 8 && m > 14)) {
        return (uint8_t) (sign | 0x7E);                 // 448, the largest finite value
    }
    return (uint8_t) (sign | ((uint32_t) (E + 7) << 3) | (uint32_t) (m - 8));
}

static __device__ __forceinline__ float fp8_gemm_to_float(const float x) { return x; }
static __device__ __forceinline__ float fp8_gemm_to_float(const half x)  { return __half2float(x); }

// one block per row: the row's power-of-two scale, its E4M3 bytes and 1/scale for the epilogue
template <typename T, int nthreads>
static __global__ void fp8_e4m3_quantize_rows(const T * __restrict__ x, uint8_t * __restrict__ q, float * __restrict__ inv_scale,
                                              const int64_t ncols) {
    const T * row  = x + blockIdx.x*ncols;
    uint8_t * qrow = q + blockIdx.x*ncols;
    float amax = 0.0f;
    for (int64_t i = threadIdx.x; i < ncols; i += nthreads) {
        amax = fmaxf(amax, fabsf(fp8_gemm_to_float(row[i])));
    }
    __shared__ __align__(16) float smax[nthreads/WARP_SIZE];
    amax = warp_reduce_max(amax);
    if (threadIdx.x % WARP_SIZE == 0) {
        smax[threadIdx.x / WARP_SIZE] = amax;
    }
    __syncthreads();
    amax = 0.0f;
#pragma unroll
    for (int w = 0; w < nthreads/WARP_SIZE; ++w) {
        amax = fmaxf(amax, smax[w]);
    }
    float s = 1.0f, inv = 1.0f;
    if (amax > 0.0f) {
        int e;
        frexpf(448.0f / amax, &e);
        s   = ldexpf(1.0f, e - 1);
        inv = ldexpf(1.0f, 1 - e);
    }
    if (threadIdx.x == 0) {
        inv_scale[blockIdx.x] = inv;
    }
    for (int64_t i = threadIdx.x; i < ncols; i += nthreads) {
        qrow[i] = fp8_e4m3_encode(fp8_gemm_to_float(row[i]) * s);
    }
}

template <typename T>
static void fp8_e4m3_quantize_rows_cuda(const T * x, uint8_t * q, float * inv_scale, const int64_t nrows, const int64_t ncols,
                                        cudaStream_t stream) {
    constexpr int nthreads = 256;
    fp8_e4m3_quantize_rows<T, nthreads><<<(unsigned) nrows, nthreads, 0, stream>>>(x, q, inv_scale, ncols);
}

#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
#define FP8_GEMM_AVAILABLE
typedef int      fp8_gemm_v2i __attribute__((ext_vector_type(2)));
typedef uint32_t fp8_gemm_u4 __attribute__((ext_vector_type(4)));   // not HIP's uint4: an array of that struct stays in scratch
typedef float    fp8_gemm_v8f __attribute__((ext_vector_type(8)));
typedef _Float16 fp8_gemm_v16h __attribute__((ext_vector_type(16)));
#endif

#if defined(FP8_GEMM_AVAILABLE) && defined(RDNA3)
// RDNA3 stand-in for the check: E4M3 widened to f16, which holds every E4M3 value exactly
static __device__ __forceinline__ _Float16 fp8_e4m3_to_f16(const uint32_t b) {
    const uint32_t e = (b >> 3) & 0xF;
    const uint32_t m = b & 7;
    uint16_t h = e == 0 ? __half_as_ushort(__float2half((float) m * (1.0f/512.0f))) : (uint16_t) (((e + 8) << 10) | (m << 7));
    h |= (uint16_t) ((b & 0x80) << 8);
    return __builtin_bit_cast(_Float16, h);
}

// the 16 K values of one WMMA from a lane's 32 bytes: bytes 0-7 and 16-23 (lo) or 8-15 and 24-31 (hi), the same K
// sets as the two gfx12 WMMAs
static __device__ __forceinline__ fp8_gemm_v16h fp8_gemm_rdna3_frag(const fp8_gemm_u4 v0, const fp8_gemm_u4 v1, const bool hi) {
    const uint32_t w[4] = { hi ? v0.z : v0.x, hi ? v0.w : v0.y, hi ? v1.z : v1.x, hi ? v1.w : v1.y };
    fp8_gemm_v16h f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        f[i] = fp8_e4m3_to_f16((w[i/4] >> (8*(i % 4))) & 0xFF);
    }
    return f;
}
#endif

// C (M x N, row stride ldc, f32) = A (M x K, E4M3, row major) * B^T (B: N x K, E4M3, the weight rows), each element
// scaled by sa[m]*sb[n]. N a multiple of BN, K of BK; rows past M are read clamped and not written.
template <int BM, int BN, int BK, int WM, int WN>
static __global__ void __launch_bounds__(WM*WN*WARP_SIZE)
fp8_gemm_tn(const uint8_t * __restrict__ A, const uint8_t * __restrict__ B, const float * __restrict__ sa,
            const float * __restrict__ sb, float * __restrict__ C, const int64_t ldc, const int M, const int N, const int K) {
#if defined(FP8_GEMM_AVAILABLE)
    constexpr int NT    = WM*WN*WARP_SIZE;
    constexpr int LDS_K = BK + 16;                 // row stride in bytes; the pad staggers the rows across the banks
    constexpr int FM    = BM/WM/16;                // 16 x 16 tiles per wave
    constexpr int FN    = BN/WN/16;
    constexpr int KC    = BK/16;                   // 16-byte chunks per tile row
    constexpr int CA    = BM*KC/NT;                // chunks per thread and K step
    constexpr int CB    = BN*KC/NT;
    static_assert(BM*KC % NT == 0 && BN*KC % NT == 0, "tile loads must split evenly over the threads");
    static_assert(BK % 32 == 0 && FM > 0 && FN > 0, "bad tile");

    __shared__ __align__(16) __attribute__((aligned(16))) uint8_t As[BM*LDS_K];
    __shared__ __align__(16) __attribute__((aligned(16))) uint8_t Bs[BN*LDS_K];

    // tiles in bands of GROUP tile rows: blocks that run at the same time read the same strips of A and B
    constexpr int GROUP = 8;
    const int nbm       = (M + BM - 1) / BM;
    const int nbn       = N / BN;
    const int per_group = GROUP*nbn;
    const int first_m   = (int) (blockIdx.x / per_group)*GROUP;
    const int gsize     = min(nbm - first_m, GROUP);
    const int in_group  = (int) (blockIdx.x % per_group);
    const int m0        = (first_m + in_group % gsize)*BM;
    const int n0        = (in_group / gsize)*BN;

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int wave = tid / WARP_SIZE;
    const int wm0  = (wave / WN)*(BM/WM);
    const int wn0  = (wave % WN)*(BN/WN);

    // staged in a native vector type: an array of HIP's uint4 (a struct around a union) stayed in scratch on gfx1201
    fp8_gemm_u4 ra[CA];
    fp8_gemm_u4 rb[CB];
#define FP8_GEMM_LOAD(k0_)                                                                              \
    _Pragma("unroll") for (int c = 0; c < CA; ++c) {                                                    \
        const int idx = tid + c*NT;                                                                     \
        ra[c] = *(const fp8_gemm_u4 *) (A + (int64_t) min(m0 + idx / KC, M - 1)*K + (k0_) + (idx % KC)*16);   \
    }                                                                                                   \
    _Pragma("unroll") for (int c = 0; c < CB; ++c) {                                                    \
        const int idx = tid + c*NT;                                                                     \
        rb[c] = *(const fp8_gemm_u4 *) (B + (int64_t) (n0 + idx / KC)*K + (k0_) + (idx % KC)*16);             \
    }
#define FP8_GEMM_COMMIT()                                                                               \
    _Pragma("unroll") for (int c = 0; c < CA; ++c) {                                                    \
        const int idx = tid + c*NT;                                                                     \
        *(fp8_gemm_u4 *) (As + (idx / KC)*LDS_K + (idx % KC)*16) = ra[c];                                     \
    }                                                                                                   \
    _Pragma("unroll") for (int c = 0; c < CB; ++c) {                                                    \
        const int idx = tid + c*NT;                                                                     \
        *(fp8_gemm_u4 *) (Bs + (idx / KC)*LDS_K + (idx % KC)*16) = rb[c];                                     \
    }

    fp8_gemm_v8f acc[FM][FN];
#pragma unroll
    for (int fm = 0; fm < FM; ++fm) {
#pragma unroll
        for (int fn = 0; fn < FN; ++fn) {
            acc[fm][fn] = fp8_gemm_v8f{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    FP8_GEMM_LOAD(0)
    FP8_GEMM_COMMIT()
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        // the next K step, in flight while this one computes. Unconditional (the last step reads its own tile again and
        // does not commit it): under a branch gfx1201 needs 183 VGPRs instead of 151 (tile 1: 256 and 40 bytes of scratch)
        FP8_GEMM_LOAD(more ? k0 + BK : k0)
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
#if defined(RDNA4)
            fp8_gemm_v2i bl[FN];
            fp8_gemm_v2i bh[FN];
#pragma unroll
            for (int fn = 0; fn < FN; ++fn) {
                const fp8_gemm_u4 v = *(const fp8_gemm_u4 *) (Bs + (wn0 + fn*16 + lane % 16)*LDS_K + kk + (lane / 16)*16);
                bl[fn] = fp8_gemm_v2i{(int) v.x, (int) v.y};
                bh[fn] = fp8_gemm_v2i{(int) v.z, (int) v.w};
            }
#pragma unroll
            for (int fm = 0; fm < FM; ++fm) {
                const fp8_gemm_u4 v = *(const fp8_gemm_u4 *) (As + (wm0 + fm*16 + lane % 16)*LDS_K + kk + (lane / 16)*16);
                const fp8_gemm_v2i al = {(int) v.x, (int) v.y};
                const fp8_gemm_v2i ah = {(int) v.z, (int) v.w};
#pragma unroll
                for (int fn = 0; fn < FN; ++fn) {
                    acc[fm][fn] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(al, bl[fn], acc[fm][fn]);
                    acc[fm][fn] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(ah, bh[fn], acc[fm][fn]);
                }
            }
#else
#pragma unroll
            for (int fm = 0; fm < FM; ++fm) {
                const uint8_t * pa = As + (wm0 + fm*16 + lane % 16)*LDS_K + kk;
                const fp8_gemm_v16h al = fp8_gemm_rdna3_frag(*(const fp8_gemm_u4 *) pa, *(const fp8_gemm_u4 *) (pa + 16), false);
                const fp8_gemm_v16h ah = fp8_gemm_rdna3_frag(*(const fp8_gemm_u4 *) pa, *(const fp8_gemm_u4 *) (pa + 16), true);
#pragma unroll
                for (int fn = 0; fn < FN; ++fn) {
                    const uint8_t * pb = Bs + (wn0 + fn*16 + lane % 16)*LDS_K + kk;
                    const fp8_gemm_u4 b0 = *(const fp8_gemm_u4 *) pb;
                    const fp8_gemm_u4 b1 = *(const fp8_gemm_u4 *) (pb + 16);
                    acc[fm][fn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, fp8_gemm_rdna3_frag(b0, b1, false), acc[fm][fn]);
                    acc[fm][fn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, fp8_gemm_rdna3_frag(b0, b1, true), acc[fm][fn]);
                }
            }
#endif // defined(RDNA4)
        }
        __syncthreads();
        if (more) {
            FP8_GEMM_COMMIT()
            __syncthreads();
        }
    }
#undef FP8_GEMM_LOAD
#undef FP8_GEMM_COMMIT

    // accumulator element g of a lane: column lane % 16 (B), row 8*(lane/16) + g on gfx12, 2*g + lane/16 on gfx11 (A)
#pragma unroll
    for (int fm = 0; fm < FM; ++fm) {
#pragma unroll
        for (int fn = 0; fn < FN; ++fn) {
            const int n = n0 + wn0 + fn*16 + lane % 16;
            const float sbn = sb[n];
#pragma unroll
            for (int g = 0; g < 8; ++g) {
#if defined(RDNA4)
                const int m = m0 + wm0 + fm*16 + 8*(lane / 16) + g;
#else
                const int m = m0 + wm0 + fm*16 + 2*g + lane / 16;
#endif // defined(RDNA4)
                if (m < M) {
                    C[(int64_t) m*ldc + n] = acc[fm][fn][g]*sa[m]*sbn;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(A, B, sa, sb, C, ldc, M, N, K);
    NO_DEVICE_CODE;
#endif // defined(FP8_GEMM_AVAILABLE)
}

template <int BM, int BN, int BK, int WM, int WN>
static void fp8_gemm_tn_cuda(const uint8_t * A, const uint8_t * B, const float * sa, const float * sb, float * C,
                             const int64_t ldc, const int M, const int N, const int K, cudaStream_t stream) {
    GGML_ASSERT(N % BN == 0 && K % BK == 0);
    const int nblocks = ((M + BM - 1) / BM)*(N / BN);
    fp8_gemm_tn<BM, BN, BK, WM, WN><<<nblocks, WM*WN*WARP_SIZE, 0, stream>>>(A, B, sa, sb, C, ldc, M, N, K);
}

// ROCMFPX_FP8_GEMM: 0 = off (the default until measured on RDNA4), 1 = on: FP8 WMMA on RDNA4, the f16 stand-in on RDNA3
static bool fp8_gemm_on(const int cc) {
    static const bool on = wmma_gemm_env("ROCMFPX_FP8_GEMM", 0) != 0;
    return on && (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));
}

// [TAG_WMMA_GEMM_SCRATCH_AUTO] How many weight rows one chunk unpacks. ROCMFPX_WMMA_GEMM_SCRATCH_MB=N: chunks of at most N MB
// of f16, as before. Unset: the whole matrix at once where that measured faster, when its unpacked copy (whole_row_bytes per
// row) fits the room the device had at its first chunked product, else 64 MB chunks (the old default). One chunk also
// writes dst directly: no output temporary, no 2D copy per chunk. Where it measured faster (RX 7900 XT, ROCm 10.1,
// test-backend-ops, Q4_K / IQ4_XS on the weight shapes of Llama-3.1-8B, Gemma-3-27B, Qwen3-32B, Mistral-24B and Qwen3.8-27B,
// with the gfx1100 table's whole-matrix entries; day-1106\scratch-ab2.ps1, one chunk against 64 MB chunks):
//   - batches of 2048+ tokens on inputs of at most 8192 (gate / up, q / o): 0.89-0.93
//   - inputs of 8192+ into outputs of 5120+ (down, Qwen3-32B's o), any batch: 0.87-0.97, Qwen3-32B's down 1.00 / 1.02
// Kept in chunks: gate / up at 512 tokens (one chunk 1.03-1.07) and Llama-3.1-8B's down (4096 outputs: 1.04-1.07). A 64 MB
// chunk fits the 7900 XT's 80 MB Infinity Cache, a whole 100-330 MB matrix does not.
// Room = cudaMemGetInfo's free memory minus 1 GiB, at most 512 MB: on Windows that figure counts this process's own
// allocations only, so the 1 GiB (llama's --fit target) is what is left for the desktop and other programs. A 20 GB card
// holding an 18.4 GB server (Qwen3.8-27B IQ4_XS at 96k, prod settings) has ~1.8 GiB of it and unpacks 17408 x 5120 (178 MB)
// in one go; a server --fit packed to its 1 GiB target keeps the chunks.
static int64_t wmma_gemm_whole_cap(const int device) {
    static std::mutex mutex;
    static int64_t cap[GGML_CUDA_MAX_DEVICES];
    static bool    known[GGML_CUDA_MAX_DEVICES] = {};
    std::lock_guard<std::mutex> lock(mutex);
    if (!known[device]) {
        size_t free = 0, total = 0;
        ggml_cuda_set_device(device);
        if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
            (void) cudaGetLastError();
            free = 0;
        }
        cap[device]   = std::clamp<int64_t>((int64_t) free - ((int64_t) 1 << 30), 0, (int64_t) 512 << 20);
        known[device] = true;
        GGML_LOG_DEBUG("%s: device %d: %lld MiB free, whole-matrix unpacks up to %lld MiB\n", __func__, device,
            (long long) (free >> 20), (long long) (cap[device] >> 20));
    }
    return cap[device];
}

static int64_t wmma_gemm_chunk_rows(const int device, const int64_t M, const int64_t N, const int64_t K,
                                    const int64_t whole_row_bytes) {
    static const int     env_mb  = wmma_gemm_env("ROCMFPX_WMMA_GEMM_SCRATCH_MB", 0);
    static const int64_t def_mb  = 64;
    const int64_t        f16_row = K*(int64_t) sizeof(half);
    const int64_t        budget  = (env_mb > 0 ? env_mb : def_mb) << 20;
    const bool whole_faster = (M >= 2048 && K <= 8192) || (N >= 5120 && K >= 8192);
    if (env_mb <= 0 && whole_faster && N*f16_row > budget && N*whole_row_bytes <= wmma_gemm_whole_cap(device)) {
        return N;
    }
    return std::min<int64_t>(N, std::max<int64_t>(256, budget / f16_row / 256 * 256));
}

static void ggml_cuda_mul_mat_fp8_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                       ggml_tensor * dst) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    cudaStream_t stream = ctx.stream();

    // activations: E4M3 with one scale per token
    ggml_cuda_pool_alloc<uint8_t> a8(ctx.pool(), M*K);
    ggml_cuda_pool_alloc<float>   sa(ctx.pool(), M);
    fp8_e4m3_quantize_rows_cuda((const float *) src1->data, a8.get(), sa.get(), M, K, stream);

    // weights: chunks of whole rows, unpacked to f16 (the existing kernels), then to E4M3 with one scale per row
    const int64_t rows = wmma_gemm_chunk_rows(ctx.device, M, N, K, K*(int64_t) (sizeof(half) + 1));
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(src0->type);
    GGML_ASSERT(to_fp16);
    ggml_cuda_pool_alloc<half>    w16(ctx.pool(), rows*K);
    ggml_cuda_pool_alloc<uint8_t> w8(ctx.pool(), rows*K);
    ggml_cuda_pool_alloc<float>   sb(ctx.pool(), rows);

    float * dst_d = (float *) dst->data;
    for (int64_t n0 = 0; n0 < N; n0 += rows) {
        const int64_t nc = std::min(rows, N - n0);
        to_fp16((const char *) src0->data + n0*src0->nb[1], w16.get(), nc*K, stream);
        fp8_e4m3_quantize_rows_cuda(w16.get(), w8.get(), sb.get(), nc, K, stream);
        // the chunk's columns straight into dst (row stride N). ROCMFPX_FP8_GEMM_TILE: 0 = 128 x 128 tiles in 8 waves of
        // 64 x 32 (the default), 1 = in 4 waves of 64 x 64. gfx1201 VGPRs per lane (compiler report): 0 fits, 1 takes
        // 253 and spills 272 bytes (16 accumulators of 8 floats plus 16 staged 16-byte loads).
        static const int tile = wmma_gemm_env("ROCMFPX_FP8_GEMM_TILE", 0);
        if (tile == 1) {
            fp8_gemm_tn_cuda<128, 128, 128, 2, 2>(a8.get(), w8.get(), sa.get(), sb.get(), dst_d + n0, N, (int) M, (int) nc,
                                                   (int) K, stream);
        } else {
            fp8_gemm_tn_cuda<128, 128, 128, 2, 4>(a8.get(), w8.get(), sa.get(), sb.get(), dst_d + n0, N, (int) M, (int) nc,
                                                   (int) K, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

bool ggml_cuda_should_use_wmma_gemm(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    static const bool    enabled   = wmma_gemm_env("ROCMFPX_WMMA_GEMM", 1) != 0;
    // at 256 tokens the unpack costs about what the GEMM saves (0.68x-1.25x by shape on gfx1100); at 512 it is 1.1x-1.4x
    static const int64_t min_batch = wmma_gemm_env("ROCMFPX_WMMA_GEMM_MIN_BATCH", 512);
    if (!enabled) {
        return false;
    }
    const bool   fp8  = fp8_gemm_on(cc);
    const char * arch = wmma_gemm_arch(cc);
    if (!fp8 && !arch) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_0_ROCMFP4:
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
        case GGML_TYPE_Q6_0_ROCMFPX:
        // [TAG_WMMA_GEMM_STOCK] stock quants, against int8 MMQ (their path on gfx1100 since TAG_MMB_RDNA3_0_MMQ), RX
        // 7900 XT 2026-10-24, 4096 x 14336 at 512 rows: Q4_0 48.1 vs 47.5 TFLOPS, Q4_K 47.2 vs 42.3, Q5_K 45.2 vs 38.7,
        // Q6_K 46.7 vs 42.8, IQ4_NL 48.0 vs 46.4, IQ4_XS 48.0 vs 46.5; Qwen3.8-27B IQ4_XS cold 17k prompt 864 vs 781 t/s
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        // [TAG_WMMA_GEMM_Q3] the 3-bit stock types (IQ3_S / IQ3_XXS of the IQ3 quants, Q3_K of Q3_K_S/M/L), unpacked to f16
        // by their existing to_fp16 kernels like the 4-bit ones above. RX 7900 XT 2026-10-26, 22016 x 7168 at 512 rows
        // against int8 MMQ: IQ3_S 55.1 vs 51.9 TFLOPS, IQ3_XXS 55.2 vs 52.7, Q3_K 51.2 vs 45.4
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_Q3_K:
        // [TAG_WMMA_GEMM_REST] the other types with a to_fp16 kernel, where it wins. RX 7900 XT 2026-10-26, 4096 x 14336 at
        // 512 rows against today's path: Q8_0 46.3 vs 44.2 TFLOPS, Q4_1 48.6 vs 41.9, Q5_1 41.6 vs 38.5, Q2_K 48.3 vs 44.3,
        // IQ2_XXS 49.2 vs 45.6, IQ2_XS 49.2 vs 44.1, IQ2_S 49.3 vs 43.9, Q3_0_ROCMFPX 37.3 vs 30.9, Q5_0_ROCMFPX 36.9 vs
        // 35.2, Q7_0_ROCMFPX 36.7 vs 35.0. Left out, slower here because their unpack costs more than the GEMM saves:
        // Q5_0 41.3 vs 45.4, Q8_0_ROCMFPX 40.8 vs 43.8, Q4_0_ROCMI4 40.6 vs 45.0; Q2_0_ROCMFPX measured unchanged.
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_Q3_0_ROCMFPX:
        case GGML_TYPE_Q5_0_ROCMFPX:
        case GGML_TYPE_Q7_0_ROCMFPX:
            break;
        default:
            return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->ne[1] < min_batch) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // the kernels read K and write N in whole tiles (at most 256 wide); token rows past M are clamped by the hardware
    if (src0->ne[0] % 256 != 0 || src0->ne[1] % 256 != 0) {
        return false;
    }
    return fp8 || wmma_gemm_load(arch) != nullptr;
}

// the f16 path into dst_d (M x N, row major); fp8_emu as ROCMFPX_WMMA_GEMM_FP8_EMU
static void wmma_gemm_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, float * dst_d,
                          const int fp8_emu) {
    const int64_t K = src0->ne[0];   // input width
    const int64_t N = src0->ne[1];   // weight rows = output width
    const int64_t M = src1->ne[1];   // tokens

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const wmma_gemm_f32_f16_t gemm = wmma_gemm_load(wmma_gemm_arch(cc));
    GGML_ASSERT(gemm);

    cudaStream_t stream = ctx.stream();

    // activations: f32 -> f16, row major M x K
    ggml_cuda_pool_alloc<half> a16(ctx.pool(), M*K);
    if (fp8_emu & 2) {
        fp8_e4m3_emulate_rows_cuda(a16.get(), (const float *) src1->data, M, K, stream);
    } else {
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, a16.get(), M*K, stream);
    }

    // weights: a chunk of whole rows at a time, unpacked to f16 into one scratch: the whole matrix where that is faster
    // and the device has room for it, else chunks of 64 MB or ROCMFPX_WMMA_GEMM_SCRATCH_MB (wmma_gemm_chunk_rows).
    // Reading the FP4 codes inside the GEMM instead (2026-10-23) was bit-exact but slower, 14-49 vs 54-67 TFLOPS:
    // every block of activation rows decodes its weight tiles again (8-22x the unpacks), where this unpacks once.
    const int64_t rows = wmma_gemm_chunk_rows(ctx.device, M, N, K, K*(int64_t) sizeof(half));
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(src0->type);
    GGML_ASSERT(to_fp16);
    ggml_cuda_pool_alloc<half> w16(ctx.pool(), rows*K);

    // a chunk's output columns are not contiguous in dst: compute them into a temporary and copy them across
    ggml_cuda_pool_alloc<float> c32(ctx.pool());
    if (rows < N) {
        c32.alloc(M*rows);
    }

    for (int64_t n0 = 0; n0 < N; n0 += rows) {
        const int64_t nc = std::min(rows, N - n0);
        to_fp16((const char *) src0->data + n0*src0->nb[1], w16.get(), nc*K, stream);
        if (fp8_emu & 1) {
            fp8_e4m3_emulate_rows_cuda(w16.get(), nullptr, nc, K, stream);
        }
        float * c = rows < N ? c32.get() : dst_d;
        // C (M x nc, row major) = A (M x K, row major) * B (K x nc, column major: the weight rows as stored)
        hipStream_t s = stream;
        gemm(c, a16.get(), w16.get(), (size_t) M, (size_t) nc, (size_t) K, 1, 0, 0, 1, &s);
        CUDA_CHECK(cudaGetLastError());
        if (rows < N) {
            CUDA_CHECK(cudaMemcpy2DAsync(dst_d + n0, N*sizeof(float), c, nc*sizeof(float), nc*sizeof(float), M,
                                         cudaMemcpyDeviceToDevice, stream));
        }
    }
}

// ROCMFPX_FP8_GEMM_CHECK=N (with ROCMFPX_FP8_GEMM=1, RDNA3 only: needs the f16 library): the first N products also run
// through the emulation (ROCMFPX_WMMA_GEMM_FP8_EMU=3) on the same inputs, and the log gets the difference. End-to-end
// numbers cannot show it: an f32 rounding difference moves some activations across an E4M3 boundary in the next layer.
static void fp8_gemm_check(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const int max_checks = wmma_gemm_env("ROCMFPX_FP8_GEMM_CHECK", 0);
    static int n_checked = 0;
    if (n_checked >= max_checks) {
        return;
    }
    ++n_checked;
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    ggml_cuda_pool_alloc<float> ref(ctx.pool(), M*N);
    wmma_gemm_f16(ctx, src0, src1, ref.get(), 3);
    std::vector<float> h_out(M*N), h_ref(M*N);
    CUDA_CHECK(cudaMemcpyAsync(h_out.data(), dst->data, M*N*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(h_ref.data(), ref.get(), M*N*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    double d2 = 0.0, r2 = 0.0, dmax = 0.0, rmax = 0.0;
    for (int64_t i = 0; i < M*N; ++i) {
        const double d = (double) h_out[i] - h_ref[i];
        d2 += d*d;
        r2 += (double) h_ref[i]*h_ref[i];
        dmax = std::max(dmax, std::fabs(d));
        rmax = std::max(rmax, (double) std::fabs(h_ref[i]));
    }
    GGML_LOG_INFO("%s: %s %s M=%lld N=%lld K=%lld: kernel vs emulation rel L2 %.3e, max |diff| %.3e of max |ref| %.3e\n",
        __func__, dst->name, ggml_type_name(src0->type), (long long) M, (long long) N, (long long) src0->ne[0],
        r2 > 0.0 ? std::sqrt(d2/r2) : 0.0, dmax, rmax);
}

void ggml_cuda_mul_mat_wmma_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (fp8_gemm_on(cc)) {
        ggml_cuda_mul_mat_fp8_gemm(ctx, src0, src1, dst);
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            fp8_gemm_check(ctx, src0, src1, dst);
        }
        return;
    }
    static const int fp8_emu = wmma_gemm_env("ROCMFPX_WMMA_GEMM_FP8_EMU", 0);
    wmma_gemm_f16(ctx, src0, src1, (float *) dst->data, fp8_emu);
}
