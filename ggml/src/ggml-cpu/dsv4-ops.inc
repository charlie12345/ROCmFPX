// ggml_compute_forward_dsv4_hc_split_sinkhorn

static void ggml_compute_forward_dsv4_hc_split_sinkhorn(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(mixes->nb[0] == sizeof(float));
    GGML_ASSERT(scale->nb[0] == sizeof(float));
    GGML_ASSERT(base->nb[0]  == sizeof(float));
    GGML_ASSERT(dst->nb[0]   == sizeof(float));

    const int n_hc           = ggml_get_op_params_i32(dst, 0);
    const int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps          = ggml_get_op_params_f32(dst, 2);
    const int64_t mix_hc     = mixes->ne[0];
    const int64_t n_rows     = ggml_nrows(mixes);

    GGML_ASSERT(n_hc > 0 && n_hc <= 16);
    GGML_ASSERT(sinkhorn_iters > 0);
    GGML_ASSERT(mix_hc == (2 + n_hc) * n_hc);
    GGML_ASSERT(ggml_nrows(dst) == n_rows);

    const float * scale_data = (const float *) scale->data;
    const float * base_data  = (const float *) base->data;

    const float pre_scale  = scale_data[0];
    const float post_scale = scale_data[1];
    const float comb_scale = scale_data[2];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t dr = (n_rows + nth - 1) / nth;
    const int64_t r0 = dr * ith;
    const int64_t r1 = std::min(r0 + dr, n_rows);

    for (int64_t r = r0; r < r1; ++r) {
        const float * mix = (const float *) ((const char *) mixes->data + r*mixes->nb[1]);
        float * out = (float *) ((char *) dst->data + r*dst->nb[1]);

        for (int i = 0; i < n_hc; ++i) {
            const float z = mix[i] * pre_scale + base_data[i];
            out[i] = 1.0f / (1.0f + expf(-z)) + eps;
        }

        for (int i = 0; i < n_hc; ++i) {
            const int off = n_hc + i;
            const float z = mix[off] * post_scale + base_data[off];
            out[off] = 2.0f / (1.0f + expf(-z));
        }

        float c[16*16];

        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float row_max = -INFINITY;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc*n_hc;
                const int off = 2*n_hc + idx;
                const float v = mix[off] * comb_scale + base_data[off];
                c[idx] = v;
                row_max = std::max(row_max, v);
            }

            float row_sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc*n_hc;
                const float v = expf(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }

            const float inv_sum = 1.0f / row_sum;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc*n_hc;
                c[idx] = c[idx] * inv_sum + eps;
            }
        }

        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc*n_hc];
            }

            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc*n_hc] *= inv_denom;
            }
        }

        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                float sum = 0.0f;
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    sum += c[src_hc + dst_hc*n_hc];
                }

                const float inv_denom = 1.0f / (sum + eps);
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    c[src_hc + dst_hc*n_hc] *= inv_denom;
                }
            }

            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                float sum = 0.0f;
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    sum += c[src_hc + dst_hc*n_hc];
                }

                const float inv_denom = 1.0f / (sum + eps);
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    c[src_hc + dst_hc*n_hc] *= inv_denom;
                }
            }
        }

        for (int i = 0; i < n_hc*n_hc; ++i) {
            out[2*n_hc + i] = c[i];
        }
    }
}

// ggml_compute_forward_dsv4_hc_weighted_sum

static void ggml_compute_forward_dsv4_hc_weighted_sum(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0]       == dst->ne[0]);
    GGML_ASSERT(x->ne[1]       == weights->ne[0]);
    GGML_ASSERT(x->ne[2]       == dst->ne[1]);
    GGML_ASSERT(weights->ne[1] == dst->ne[1]);
    GGML_ASSERT(x->ne[3]       == 1);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(weights->ne[3] == 1);
    GGML_ASSERT(dst->ne[2]     == 1);
    GGML_ASSERT(dst->ne[3]     == 1);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = x->ne[1];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_elem   = n_embd * n_tokens;

    const int64_t i0 = (n_elem * params->ith) / params->nth;
    const int64_t i1 = (n_elem * (params->ith + 1)) / params->nth;

    const char * x_data = (const char *) x->data;
    const char * w_data = (const char *) weights->data;
          char * y_data = (      char *) dst->data;

    for (int64_t i = i0; i < i1; ++i) {
        const int64_t d = i % n_embd;
        const int64_t t = i / n_embd;

        float acc = 0.0f;
        for (int64_t h = 0; h < n_hc; ++h) {
            const float xv = *(const float *) (x_data + d*x->nb[0] + h*x->nb[1] + t*x->nb[2]);
            const float wv = *(const float *) (w_data + h*weights->nb[0] + t*weights->nb[1]);
            acc += xv * wv;
        }

        *(float *) (y_data + d*dst->nb[0] + t*dst->nb[1]) = acc;
    }
}

// ggml_compute_forward_dsv4_hc_expand

static void ggml_compute_forward_dsv4_hc_expand(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);
    GGML_ASSERT(block_out->ne[0] == dst->ne[0]);
    GGML_ASSERT(block_out->ne[1] == dst->ne[2]);
    GGML_ASSERT(residual->ne[0]  == dst->ne[0]);
    GGML_ASSERT(residual->ne[1]  == dst->ne[1]);
    GGML_ASSERT(residual->ne[2]  == dst->ne[2]);
    GGML_ASSERT(post->ne[0]      == dst->ne[1]);
    GGML_ASSERT(post->ne[1]      == dst->ne[2]);
    GGML_ASSERT(comb->ne[0]      == dst->ne[1]);
    GGML_ASSERT(comb->ne[1]      == dst->ne[1]);
    GGML_ASSERT(comb->ne[2]      == dst->ne[2]);
    GGML_ASSERT(block_out->ne[3] == 1);
    GGML_ASSERT(residual->ne[3]  == 1);
    GGML_ASSERT(post->ne[2]      == 1);
    GGML_ASSERT(post->ne[3]      == 1);
    GGML_ASSERT(comb->ne[3]      == 1);
    GGML_ASSERT(dst->ne[3]       == 1);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = dst->ne[1];
    const int64_t n_tokens = dst->ne[2];
    const int64_t n_elem   = n_embd * n_hc * n_tokens;

    const int64_t i0 = (n_elem * params->ith) / params->nth;
    const int64_t i1 = (n_elem * (params->ith + 1)) / params->nth;

    const char * block_data = (const char *) block_out->data;
    const char * res_data   = (const char *) residual->data;
    const char * post_data  = (const char *) post->data;
    const char * comb_data  = (const char *) comb->data;
          char * dst_data   = (      char *) dst->data;

    for (int64_t i = i0; i < i1; ++i) {
        const int64_t d      = i % n_embd;
        const int64_t tmp    = i / n_embd;
        const int64_t dst_hc = tmp % n_hc;
        const int64_t t      = tmp / n_hc;

        const float block_v = *(const float *) (block_data + d*block_out->nb[0] + t*block_out->nb[1]);
        const float post_v  = *(const float *) (post_data  + dst_hc*post->nb[0] + t*post->nb[1]);

        float acc = block_v * post_v;
        for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
            const float comb_v = *(const float *) (comb_data + dst_hc*comb->nb[0] + src_hc*comb->nb[1] + t*comb->nb[2]);
            const float res_v  = *(const float *) (res_data  + d*residual->nb[0] + src_hc*residual->nb[1] + t*residual->nb[2]);
            acc += comb_v * res_v;
        }

        *(float *) (dst_data + d*dst->nb[0] + dst_hc*dst->nb[1] + t*dst->nb[2]) = acc;
    }
}

static float ggml_dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = std::min(std::fabs(x), 448.0f);

    int best = 0;
    float best_diff = ax;

    for (int i = 1; i < 127; ++i) {
        const int exp  = (i >> 3) & 0x0f;
        const int mant = i & 0x07;
        const float val = exp == 0
            ? std::ldexp(float(mant), -9)
            : std::ldexp(1.0f + float(mant) / 8.0f, exp - 7);
        const float diff = std::fabs(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }

    const int exp  = (best >> 3) & 0x0f;
    const int mant = best & 0x07;
    const float val = exp == 0
        ? std::ldexp(float(mant), -9)
        : std::ldexp(1.0f + float(mant) / 8.0f, exp - 7);

    return sign * val;
}

static void ggml_compute_forward_dsv4_fp8_kv_quantize(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t n_rot = ggml_get_op_params_i32(dst, 0);
    const int64_t head_dim = src0->ne[0];
    const int64_t n_nope = head_dim - n_rot;

    GGML_ASSERT(n_rot >= 0);
    GGML_ASSERT(n_nope > 0);
    GGML_ASSERT(n_nope % 64 == 0);

    const int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int64_t row_start = (n_rows * params->ith) / params->nth;
    const int64_t row_end   = (n_rows * (params->ith + 1)) / params->nth;

    for (int64_t row = row_start; row < row_end; ++row) {
        const int64_t i1 = row % src0->ne[1];
        const int64_t i2 = (row / src0->ne[1]) % src0->ne[2];
        const int64_t i3 = row / (src0->ne[1] * src0->ne[2]);

        const char * src_base = (const char *) src0->data + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
        char       * dst_base = (      char *) dst->data  + i1*dst->nb[1]  + i2*dst->nb[2]  + i3*dst->nb[3];

        for (int64_t off = 0; off < n_nope; off += 64) {
            float amax = 0.0f;
            for (int64_t i = 0; i < 64; ++i) {
                const float v = *(const float *) (src_base + (off + i)*src0->nb[0]);
                amax = std::max(amax, std::fabs(v));
            }

            amax = std::max(amax, 1.0e-4f);
            const float scale = std::ldexp(1.0f, int(std::ceil(std::log2(amax / 448.0f))));
            for (int64_t i = 0; i < 64; ++i) {
                const float v = *(const float *) (src_base + (off + i)*src0->nb[0]);
                *(float *) (dst_base + (off + i)*dst->nb[0]) =
                    ggml_dsv4_e4m3fn_dequant(std::clamp(v / scale, -448.0f, 448.0f)) * scale;
            }
        }

        for (int64_t i = n_nope; i < head_dim; ++i) {
            *(float *) (dst_base + i*dst->nb[0]) = *(const float *) (src_base + i*src0->nb[0]);
        }
    }
}
