#include "dsa-mask.cuh"

#include <cmath>
#include <cstdlib>

// GLM-5.3-Flash DSA mask: the graph builds the attention mask of the indexer selection as
//   ADD(VIEW(SET_ROWS(dst = RESHAPE(REPEAT(FILL(-inf))), rows = RESHAPE(REPEAT(FILL(0))), idx)), kq_mask)
// i.e. the causal mask value at each selected cell and -inf elsewhere. The chain streams the full
// [n_kv + n_sel, n_tokens] f16 tensor several times (about 1 GB per pass for a 4096-token ubatch at 128K). Here the
// ADD's output is written directly: -inf everywhere, then the causal mask value at each selected cell. Bit-identical
// (in f16, -inf + x = -inf and 0 + x = x); dead slots point at dump rows >= n_kv, which the view drops.
// The REPEATs and the SET_ROWS are skipped; nothing else may read them. GGML_CUDA_DSA_MASK_FUSE=0 disables it.

static bool dsa_fuse_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_DSA_MASK_FUSE");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

static int dsa_node_index(const ggml_cgraph * cgraph, const ggml_tensor * t, const int hint) {
    for (int n = hint; n >= 0; --n) {
        if (cgraph->nodes[n] == t) {
            return n;
        }
    }
    return -1;
}

static bool dsa_used_once(const ggml_cgraph * cgraph, const ggml_tensor * t, const int hint) {
    const int n = dsa_node_index(cgraph, t, hint);
    return n >= 0 && ggml_node_get_use_count(cgraph, n) == 1;
}

// t = RESHAPE(REPEAT(FILL(value))) with every link read once; returns the REPEAT
static const ggml_tensor * dsa_fill_repeat(const ggml_cgraph * cgraph, const ggml_tensor * t, const float value, const int hint) {
    if (!t || t->op != GGML_OP_RESHAPE || !dsa_used_once(cgraph, t, hint)) {
        return nullptr;
    }
    const ggml_tensor * r = t->src[0];
    if (!r || r->op != GGML_OP_REPEAT || r->type != GGML_TYPE_F16 || !ggml_is_contiguous(r) || !dsa_used_once(cgraph, r, hint)) {
        return nullptr;
    }
    const ggml_tensor * f = r->src[0];
    if (!f || f->op != GGML_OP_FILL || f->type != GGML_TYPE_F16) {
        return nullptr;
    }
    const float c = ggml_get_op_params_f32(f, 0);
    const bool ok = std::isinf(value) ? (std::isinf(c) && c < 0.0f) : c == value;
    return ok ? r : nullptr;
}

bool ggml_cuda_dsa_mask_match_at_add(const ggml_cgraph * cgraph, const int i, ggml_cuda_dsa_mask_match & m) {
    if (!dsa_fuse_enabled()) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[i];
    if (add->op != GGML_OP_ADD || add->type != GGML_TYPE_F16 || !ggml_is_contiguous(add) || add->ne[2] != 1 || add->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * v = add->src[0], * km = add->src[1];
    if (!v || !km || v->op != GGML_OP_VIEW || v->view_offs != 0 || !dsa_used_once(cgraph, v, i)) {
        return false;
    }
    const ggml_tensor * s = v->src[0];
    if (!s || s->op != GGML_OP_SET_ROWS || s->type != GGML_TYPE_F16 || !dsa_used_once(cgraph, s, i)) {
        return false;
    }
    const ggml_tensor * rows = s->src[0], * idx = s->src[1], * dst = s->src[2];
    const ggml_tensor * rep_inf  = dsa_fill_repeat(cgraph, dst,  -INFINITY, i);
    const ggml_tensor * rep_zero = dsa_fill_repeat(cgraph, rows, 0.0f, i);
    if (!rep_inf || !rep_zero || !idx || idx->type != GGML_TYPE_I32 || !ggml_is_contiguous(idx)) {
        return false;
    }
    const int64_t n_kv = add->ne[0], n_tok = add->ne[1];
    // dst [1, n_kv + n_sel, n_tokens], idx [n_sel, n_tokens], the add reads the first n_kv of each token's row
    const int64_t n_sel = idx->ne[0];
    if (dst->ne[0] != 1 || dst->ne[1] != n_kv + n_sel || dst->ne[2] != n_tok || idx->ne[1] != n_tok ||
            v->ne[0] != n_kv || v->ne[1] != n_tok || v->nb[1] != dst->nb[2]) {
        return false;
    }
    if (km->type != GGML_TYPE_F16 || km->ne[0] != n_kv || km->ne[1] != n_tok || km->nb[0] != sizeof(half)) {
        return false;
    }
    // the output is written before the indices and the causal mask are read: they must not share its memory (the
    // allocator may reuse the indices' buffer, whose last reader would have been the skipped SET_ROWS)
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    if (overlap(add, idx) || overlap(add, km)) {
        return false;
    }
    m.add = add; m.idx = idx; m.kq_mask = km; m.rep_inf = rep_inf; m.rep_zero = rep_zero; m.set_rows = s;
    m.n_kv = n_kv; m.n_sel = n_sel; m.n_tok = n_tok;
    return true;
}

bool ggml_cuda_dsa_mask_part_skippable(const ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * t = cgraph->nodes[i];
    if (!dsa_fuse_enabled() || (t->op != GGML_OP_REPEAT && t->op != GGML_OP_SET_ROWS)) {
        return false;
    }
    for (int n = i + 1; n < cgraph->n_nodes && n < i + 512; ++n) {
        if (cgraph->nodes[n]->op != GGML_OP_ADD) {
            continue;
        }
        ggml_cuda_dsa_mask_match m;
        if (ggml_cuda_dsa_mask_match_at_add(cgraph, n, m) && (m.rep_inf == t || m.rep_zero == t || m.set_rows == t)) {
            return true;
        }
    }
    return false;
}

static __global__ void dsa_mask_fill(half * __restrict__ out, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(-INFINITY);
    }
}

static __global__ void dsa_mask_scatter(half * __restrict__ out, const half * __restrict__ km, const int32_t * __restrict__ idx,
        const int n_kv, const int n_sel, const int64_t s_km, const int64_t s_idx) {
    const int t = blockIdx.y;
    for (int s = blockIdx.x*blockDim.x + threadIdx.x; s < n_sel; s += gridDim.x*blockDim.x) {
        const int c = idx[(int64_t) t*s_idx + s];
        if (c >= 0 && c < n_kv) {
            out[(int64_t) t*n_kv + c] = km[(int64_t) t*s_km + c];
        }
    }
}

// Per token, the attention's list of the cells the mask leaves visible (selected, below n_kv, causal value finite),
// in ascending order and padded with -1 to n_sel, as ggml_cuda_flash_attn_ext_compact_mask produces it from the
// mask; the glm512 attention takes it from here instead of scanning the mask again. One block per token sorts its
// entries in LDS (bitonic). Lists stay valid until a node writes the mask's memory or the graph ends.
#define DSA_LIST_WORDS 8192   // bitmap words: caches up to 262144 cells

// token t's selected visible cells, ascending, padded with -1 to n_sel (what the attention's mask compaction
// produces from the dense mask, without reading it): the cells are set in an LDS bitmap over the cache, then each
// thread writes the set bits of its run of words at the exclusive prefix of the per-thread counts
static __global__ void __launch_bounds__(256) dsa_mask_lists(const half * __restrict__ km, const int32_t * __restrict__ idx,
        int32_t * __restrict__ lists, int32_t * __restrict__ counts, const int n_kv, const int n_sel, const int64_t s_km,
        const int64_t s_idx) {
    __shared__ uint32_t bm[DSA_LIST_WORDS];
    __shared__ int pre[256];
    const int t = blockIdx.x, tid = threadIdx.x;
    const int nw = (n_kv + 31)/32;
    for (int i = tid; i < nw; i += 256) {
        bm[i] = 0;
    }
    __syncthreads();
    for (int s = tid; s < n_sel; s += 256) {
        const int c = idx[(int64_t) t*s_idx + s];
        if (c >= 0 && c < n_kv && !__hisinf(km[(int64_t) t*s_km + c])) {
            atomicOr(&bm[c/32], 1u << (c % 32));
        }
    }
    __syncthreads();
    const int per = (nw + 255)/256;
    const int w0 = min(tid*per, nw), w1 = min(w0 + per, nw);
    int cnt = 0;
    for (int w = w0; w < w1; ++w) {
        cnt += __popc(bm[w]);
    }
    pre[tid] = cnt;
    __syncthreads();
    for (int off = 1; off < 256; off *= 2) {
        const int x = tid >= off ? pre[tid - off] : 0;
        __syncthreads();
        pre[tid] += x;
        __syncthreads();
    }
    int32_t * out = lists + (int64_t) t*n_sel;
    int o = pre[tid] - cnt;
    for (int w = w0; w < w1; ++w) {
        uint32_t bits = bm[w];
        while (bits) {
            const int k = __ffs(bits) - 1;
            out[o++] = 32*w + k;
            bits &= bits - 1;
        }
    }
    const int count = pre[255];
    for (int i = count + tid; i < n_sel; i += 256) {
        out[i] = -1;
    }
    if (tid == 0) {
        counts[t] = count;
    }
}

struct dsa_lists_slot {
    int32_t *    buf  = nullptr;   // lists [n_tok][n_sel], then counts [n_tok]
    size_t       size = 0;         // ints
    const void * mask = nullptr;
    size_t       mask_bytes = 0;
    size_t       mask_nb1 = 0;
    int64_t      n_tok = 0, n_sel = 0;
    bool         valid = false;
};
static dsa_lists_slot g_dsa_lists[GGML_CUDA_MAX_DEVICES];

void ggml_cuda_dsa_lists_reset(const int device) {
    g_dsa_lists[device].valid = false;
}

void ggml_cuda_dsa_lists_invalidate(const int device, const ggml_tensor * t) {
    dsa_lists_slot & s = g_dsa_lists[device];
    if (!s.valid || !t || !t->data) {
        return;
    }
    const char * a = (const char *) t->data, * m = (const char *) s.mask;
    if (a < m + s.mask_bytes && m < a + ggml_nbytes(t)) {
        s.valid = false;
    }
}

bool ggml_cuda_dsa_lists_lookup(const int device, const ggml_tensor * mask, const int n_q, const int n_kv_max,
        const int32_t ** lists, const int32_t ** counts) {
    const dsa_lists_slot & s = g_dsa_lists[device];
    if (!s.valid || mask->data != s.mask || mask->nb[1] != s.mask_nb1 || n_kv_max != s.n_sel || n_q > s.n_tok) {
        return false;
    }
    *lists  = s.buf;
    *counts = s.buf + s.n_tok*s.n_sel;
    return true;
}

void ggml_cuda_dsa_mask(ggml_backend_cuda_context & ctx, const ggml_cuda_dsa_mask_match & m) {
    cudaStream_t stream = ctx.stream();
    half * out = (half *) m.add->data;
    const int64_t n = m.n_kv*m.n_tok;
    dsa_mask_fill<<<(unsigned) ((n + 255)/256), 256, 0, stream>>>(out, n);
    const dim3 grid((unsigned) std::min<int64_t>((m.n_sel + 255)/256, 16), (unsigned) m.n_tok, 1);
    dsa_mask_scatter<<<grid, 256, 0, stream>>>(out, (const half *) m.kq_mask->data, (const int32_t *) m.idx->data,
        (int) m.n_kv, (int) m.n_sel, m.kq_mask->nb[1]/sizeof(half), m.idx->nb[1]/sizeof(int32_t));
    CUDA_CHECK(cudaGetLastError());

    // the attention's per-token lists, built from the indices instead of a scan of the mask (main stream only)
    dsa_lists_slot & sl = g_dsa_lists[ctx.device];
    sl.valid = false;
    static const bool lists_on = [] {
        const char * e = getenv("GGML_CUDA_DSA_LISTS");
        return e == nullptr || atoi(e) != 0;
    }();
    if (lists_on && ctx.curr_stream_no == 0 && (m.n_kv + 31)/32 <= DSA_LIST_WORDS) {
        const size_t need = (size_t) m.n_tok*m.n_sel + m.n_tok;
        if (sl.size < need) {
            void * p = nullptr;
            if (cudaMalloc(&p, need*sizeof(int32_t)) != cudaSuccess) {
                (void) cudaGetLastError();
                return;
            }
            sl.buf  = (int32_t *) p;   // an outgrown buffer is not freed: a captured graph may still use it
            sl.size = need;
        }
        dsa_mask_lists<<<(unsigned) m.n_tok, 256, 0, stream>>>((const half *) m.kq_mask->data, (const int32_t *) m.idx->data,
            sl.buf, sl.buf + m.n_tok*m.n_sel, (int) m.n_kv, (int) m.n_sel, m.kq_mask->nb[1]/sizeof(half), m.idx->nb[1]/sizeof(int32_t));
        CUDA_CHECK(cudaGetLastError());
        sl.mask = m.add->data; sl.mask_bytes = ggml_nbytes(m.add); sl.mask_nb1 = m.add->nb[1];
        sl.n_tok = m.n_tok; sl.n_sel = m.n_sel; sl.valid = true;
    }
}
