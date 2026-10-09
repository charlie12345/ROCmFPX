#include "gdn-conv.cuh"
#include "unary.cuh"
#include <cstdlib>

static bool gdn_conv_enabled() { return true; }

static bool gdn_is_view_op(const ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_TRANSPOSE || op == GGML_OP_PERMUTE || op == GGML_OP_NONE;
}

// tail-only materialization of the concat: columns [tail_from, T+3) of every channel row
static __global__ void gdn_concat_tail(const float * __restrict__ state, const float * __restrict__ x, float * __restrict__ out,
                                       const int C, const int T, const int tail_from, const int row_stride) {
    const int ncols = T + 3 - tail_from;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) C * ncols) return;
    const int c = (int) (idx / ncols), j = tail_from + (int) (idx % ncols);
    out[(size_t) c * row_stride + j] = (j < 3) ? state[c * 3 + j] : x[(size_t) (j - 3) * C + c];
}

template <bool apply_silu, int TT>
static __global__ void __launch_bounds__(256) gdn_conv_direct_kernel(const float * __restrict__ state, const float * __restrict__ x,
        const float * __restrict__ w, float * __restrict__ y, const int C, const int T) {
    const int c  = blockIdx.x * 256 + threadIdx.x;
    const int t0 = blockIdx.y * TT;
    if (c >= C) return;
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wr[j] = w[c * 4 + j];
    float xw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) { const int jp = t0 + k; xw[k] = (jp < 3) ? state[c * 3 + jp] : x[(size_t) (jp - 3) * C + c]; }
    const float b = 0.0f;
    const int tend = min(T, t0 + TT);
    for (int t = t0; t < tend; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumf += xw[j] * wr[j];
        sumf += b;
        y[(size_t) t * C + c] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
        const int jp = t + 4; xw[3] = (jp < 3) ? state[c * 3 + jp] : ((jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f);
    }
}

static bool gdn_conv_check(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) {
    if (!gdn_conv_enabled() || i < 0 || i + 2 >= cgraph->n_nodes) return false;
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0) return false;
    const ggml_tensor * st = cc->src[0]; const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32) return false;
    // the transposed input may be materialized by a single-use CONT, which the fused path then skips
    if (tr->op == GGML_OP_CONT && tr->src[0] && tr->src[0]->op == GGML_OP_TRANSPOSE) {
        int ci = -1;
        for (int n = i - 1; n >= 0 && n >= i - 8; --n) if (cgraph->nodes[n] == tr) { ci = n; break; }
        if (ci < 0 || !ggml_node_has_n_uses(cgraph, ci, 1)) return false;
        for (int n = ci + 1; n < i; ++n) if (!gdn_is_view_op(cgraph->nodes[n]->op)) return false;
        m.cont_idx = ci;
        tr = tr->src[0];
    } else {
        m.cont_idx = -1;
    }
    if (tr->op != GGML_OP_TRANSPOSE || !tr->view_src) return false;
    const ggml_tensor * x = tr->view_src;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) return false;
    const int64_t C = x->ne[0], T = x->ne[1];
    if (st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != 1 || st->ne[3] != 1 || !ggml_is_contiguous(st)) return false;
    if (tr->ne[0] != T || tr->ne[1] != C || tr->view_offs != 0) return false;
    if (cc->ne[0] != T + 3 || cc->ne[1] != C || cc->ne[2] != 1 || cc->ne[3] != 1 || !ggml_is_contiguous(cc)) return false;
    if (T < 256 || C % 256 != 0) return false;
    if (i + 1 >= cgraph->n_nodes || cgraph->nodes[i + 1]->op != GGML_OP_VIEW || cgraph->nodes[i + 1]->view_src != cc) return false;
    int64_t tail_from = T + 3; int conv = -1;
    for (int n = i + 1; n < cgraph->n_nodes && n < i + 64; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->op == GGML_OP_SSM_CONV && t->src[0] == cc) { conv = n; break; }
        bool reads = false;
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == cc || t->src[s]->view_src == cc) reads = true;
        if (!reads) continue;
        if (t->op == GGML_OP_VIEW) {
            if (t->ne[0] != 3 || t->ne[1] != C || t->nb[1] != cc->nb[1] || t->view_offs % sizeof(float)) return false;
            const int64_t col = (int64_t) (t->view_offs / sizeof(float)); if (col < 0 || col + 3 > T + 3) return false;
            tail_from = std::min(tail_from, col); continue;
        }
        if ((t->op == GGML_OP_CONT || t->op == GGML_OP_CPY) && t->src[0]->op == GGML_OP_VIEW && t->src[0]->view_src == cc) continue;
        return false;
    }
    if (conv < 0) return false;
    const ggml_tensor * cv = cgraph->nodes[conv];
    const ggml_tensor * w = cv->src[1];
    if (cv->type != GGML_TYPE_F32 || !w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C || !ggml_is_contiguous(w)) return false;
    if (cv->ne[0] != C || cv->ne[1] != T || cv->ne[2] != 1 || !ggml_is_contiguous(cv)) return false;
    if (conv + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * su = cgraph->nodes[conv + 1];
    if (su->op != GGML_OP_UNARY || ggml_get_unary_op(su) != GGML_UNARY_OP_SILU || su->src[0] != cv || su->type != GGML_TYPE_F32 || !ggml_is_contiguous(su)) return false;
    // nothing after the conv may read the conv output or the (only partially written) concat
    for (int n = conv + 2; n < cgraph->n_nodes; ++n) { const ggml_tensor * t = cgraph->nodes[n];
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
            const ggml_tensor * r = t->src[s];
            if (r == cv || r->view_src == cv || r == cc || r->view_src == cc) return false;
        } }
    m.concat_idx = i; m.conv_idx = conv; m.x = x; m.state = st; m.concat = cc; m.w = w; m.conv_out = cgraph->nodes[conv + 1];
    m.C = C; m.T = T; m.tail_from = tail_from; m.silu = true;
    return true;
}

bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) { return gdn_conv_check(cgraph, i, m); }

bool ggml_cuda_gdn_conv_match_at_cont(const ggml_cgraph * cgraph, int k, ggml_cuda_gdn_conv_match & m) {
    if (!gdn_conv_enabled() || cgraph->nodes[k]->op != GGML_OP_CONT) return false;
    for (int i = k + 1; i < cgraph->n_nodes && i <= k + 8; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (t->op == GGML_OP_CONCAT && t->src[1] == cgraph->nodes[k]) return gdn_conv_check(cgraph, i, m) && m.cont_idx == k;
        if (!gdn_is_view_op(t->op)) return false;
    }
    return false;
}

bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m) {
    if (!gdn_conv_enabled() || j < 1 || cgraph->nodes[j]->op != GGML_OP_SSM_CONV) return false;
    const ggml_tensor * cc = cgraph->nodes[j]->src[0];
    for (int i = j - 1; i >= 0 && i >= j - 64; --i) if (cgraph->nodes[i] == cc) return gdn_conv_check(cgraph, i, m) && m.conv_idx == j;
    return false;
}

void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    const int ncols = (int) (m.T + 3 - m.tail_from);
    if (ncols <= 0) return;
    const int64_t n = m.C * ncols;
    gdn_concat_tail<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (float *) m.concat->data, (int) m.C, (int) m.T, (int) m.tail_from, (int) (m.concat->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    constexpr int TT = 128;
    dim3 grid((unsigned) (m.C / 256), (unsigned) ((m.T + TT - 1) / TT));
    gdn_conv_direct_kernel<true, TT><<<grid, 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (const float *) m.w->data, (float *) m.conv_out->data, (int) m.C, (int) m.T);
    CUDA_CHECK(cudaGetLastError());
}

// Decode and short batches (T <= 32 tokens, e.g. speculative verification): CONCAT(state, x^T), the conv-state CPYs
// out of the concat and SSM_CONV + SILU as one kernel. Each channel slides its window over the T new values, writes
// the SiLU conv outputs and copies the saved windows straight into the CPY destinations; the concat is never
// materialized (and a CONT feeding it, if any, is simply not read).
static __global__ void __launch_bounds__(256) gdn_conv_decode_kernel(const float * __restrict__ state, const float * __restrict__ x,
        const float * __restrict__ w, float * __restrict__ y, const ggml_cuda_gdn_conv_decode_match::cpy_args cp, const int C,
        const int T) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    // concat column j of this channel: the 3 state values, then the new values
    auto col = [&](const int j) { return j < 3 ? state[c * 3 + j] : x[(size_t) (j - 3) * C + c]; };
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wr[j] = w[c * 4 + j];
    float w0 = col(0), w1 = col(1), w2 = col(2);
    for (int t = 0; t < T; ++t) {
        const float w3 = col(t + 3);
        const float sumf = fmaf(w0, wr[0], fmaf(w1, wr[1], fmaf(w2, wr[2], w3 * wr[3])));
        y[(size_t) t * C + c] = ggml_cuda_op_silu_single(sumf);
        w0 = w1; w1 = w2; w2 = w3;
    }
#pragma unroll
    for (int k = 0; k < GDN_CONV_DECODE_MAX_CPY; ++k) {
        if (k < cp.n) {
            const int s = cp.col[k];
            float * d = cp.dst[k] + c * 3;
            d[0] = col(s + 0);
            d[1] = col(s + 1);
            d[2] = col(s + 2);
        }
    }
}

bool ggml_cuda_gdn_conv_decode_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_decode_match & m) {
    if (!gdn_conv_enabled() || i < 0 || i + 2 >= cgraph->n_nodes) return false;
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0) return false;
    const ggml_tensor * st = cc->src[0]; const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32) return false;
    // the transposed input, possibly made contiguous by a CONT whose result is then not needed
    if (tr->op == GGML_OP_CONT && tr->src[0] && tr->src[0]->op == GGML_OP_TRANSPOSE) tr = tr->src[0];
    if (tr->op != GGML_OP_TRANSPOSE || !tr->view_src) return false;
    const ggml_tensor * x = tr->view_src;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || tr->view_offs != 0) return false;
    const int64_t C = st->ne[1], T = cc->ne[0] - 3;
    if (T < 1 || T > 32) return false;
    if (st->ne[0] != 3 || st->ne[2] != 1 || st->ne[3] != 1 || !ggml_is_contiguous(st)) return false;
    if (tr->ne[0] != T || tr->ne[1] != C || ggml_nelements(x) != C * T || x->ne[0] != C) return false;
    if (cc->ne[1] != C || cc->ne[2] != 1 || cc->ne[3] != 1 || !ggml_is_contiguous(cc)) return false;
    m.cp.n = 0;
    int conv = -1;
    for (int n = i + 1; n < cgraph->n_nodes && n < i + 64; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->op == GGML_OP_SSM_CONV && t->src[0] == cc) { conv = n; break; }
        if (gdn_is_view_op(t->op)) continue;
        // only the conv-state saves may sit in between: 3-column windows of the concat into contiguous [3, C] views
        if (t->op != GGML_OP_CPY || m.cp.n == GDN_CONV_DECODE_MAX_CPY) return false;
        const ggml_tensor * v = t->src[0]; const ggml_tensor * d = t->src[1];
        if (!v || v->op != GGML_OP_VIEW || v->view_src != cc || v->ne[0] != 3 || v->ne[1] != C || v->nb[1] != cc->nb[1]) return false;
        if (v->view_offs % sizeof(float) || (int64_t) (v->view_offs / sizeof(float)) > T || ggml_nelements(v) != 3 * C) return false;
        if (!d || d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || ggml_nelements(d) != 3 * C) return false;
        m.cp.dst[m.cp.n] = (float *) d->data;
        m.cp.col[m.cp.n] = (int) (v->view_offs / sizeof(float));
        m.cp.n++;
    }
    if (conv < 0 || conv + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * cv = cgraph->nodes[conv];
    const ggml_tensor * w = cv->src[1];
    if (cv->type != GGML_TYPE_F32 || !w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C || !ggml_is_contiguous(w)) return false;
    if (cv->ne[0] != C || cv->ne[1] != T || cv->ne[2] != 1) return false;
    const ggml_tensor * su = cgraph->nodes[conv + 1];
    if (su->op != GGML_OP_UNARY || ggml_get_unary_op(su) != GGML_UNARY_OP_SILU || su->src[0] != cv || su->type != GGML_TYPE_F32 || !ggml_is_contiguous(su)) return false;
    // nothing after may read the concat, the CPY results or the raw conv output
    for (int n = conv + 2; n < cgraph->n_nodes; ++n) { const ggml_tensor * t = cgraph->nodes[n];
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
            const ggml_tensor * r = t->src[s];
            if (r == cv || r->view_src == cv || r == cc || r->view_src == cc) return false;
        } }
    m.concat_idx = i; m.silu_idx = conv + 1; m.x = x; m.state = st; m.w = w; m.out = cgraph->nodes[conv + 1]; m.C = C; m.T = T;
    return true;
}

void ggml_cuda_gdn_conv_decode(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_decode_match & m) {
    gdn_conv_decode_kernel<<<(unsigned) ((m.C + 255) / 256), 256, 0, ctx.stream()>>>((const float *) m.state->data,
        (const float *) m.x->data, (const float *) m.w->data, (float *) m.out->data, m.cp, (int) m.C, (int) m.T);
    CUDA_CHECK(cudaGetLastError());
}
