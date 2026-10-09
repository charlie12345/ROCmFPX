// requant-sidecar: requantize selected tensors of a (split) GGUF model into a small sidecar GGUF. The loader applies it
// with LLAMA_WEIGHT_OVERRIDE=<sidecar>, so the original files stay untouched and a variant costs only its changed
// tensors on disk.
//
// usage: llama-requant-sidecar [--imatrix <imatrix.gguf>] <out.gguf> <dst-type> <src-type|any> <name-regex>
//                               <model.gguf> [more splits...]
// With an imatrix (llama-imatrix GGUF output), each column is weighted by its mean squared activation, as
// llama-quantize does.

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <thread>
#include <vector>

static ggml_type type_from_name(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcmp(n, s) == 0) {
            return (ggml_type) t;
        }
    }
    return GGML_TYPE_COUNT;
}

int main(int argc, char ** argv) {
    // optional importance matrix: column weights in_sum2 / counts per tensor
    ggml_context * ictx = nullptr;
    gguf_context * imat = nullptr;
    if (argc > 2 && strcmp(argv[1], "--imatrix") == 0) {
        gguf_init_params ig = { false, &ictx };
        imat = gguf_init_from_file(argv[2], ig);
        if (!imat) {
            fprintf(stderr, "cannot read imatrix %s\n", argv[2]);
            return 1;
        }
        argv += 2;
        argc -= 2;
    }
    auto column_weights = [&](const char * name, const int64_t n_per_row, std::vector<float> & w) -> bool {
        if (!ictx) {
            return false;
        }
        const ggml_tensor * s2 = ggml_get_tensor(ictx, (std::string(name) + ".in_sum2").c_str());
        const ggml_tensor * ct = ggml_get_tensor(ictx, (std::string(name) + ".counts").c_str());
        if (!s2 || !ct || ggml_nelements(s2) != n_per_row || s2->type != GGML_TYPE_F32 || ct->type != GGML_TYPE_F32) {
            return false;
        }
        const float n = ((const float *) ct->data)[0];
        if (n <= 0.0f) {
            return false;
        }
        w.resize(n_per_row);
        for (int64_t i = 0; i < n_per_row; ++i) {
            w[i] = ((const float *) s2->data)[i] / n;
        }
        return true;
    };
    if (argc < 6) {
        fprintf(stderr, "usage: %s <out.gguf> <dst-type> <src-type|any> <name-regex> <model.gguf> [more splits...]\n", argv[0]);
        return 1;
    }
    const char * out_path = argv[1];
    const ggml_type dst = type_from_name(argv[2]);
    const bool any_src = strcmp(argv[3], "any") == 0;
    const ggml_type src_filter = any_src ? GGML_TYPE_COUNT : type_from_name(argv[3]);
    if (dst == GGML_TYPE_COUNT || (!any_src && src_filter == GGML_TYPE_COUNT)) {
        fprintf(stderr, "unknown type\n");
        return 1;
    }
    const std::regex re(argv[4]);
    ggml_quantize_init(dst);

    gguf_context * out = gguf_init_empty();
    gguf_set_val_str(out, "general.name", "weight override sidecar");
    gguf_set_val_str(out, "requant_sidecar.dst_type", ggml_type_name(dst));
    gguf_set_val_str(out, "requant_sidecar.pattern", argv[4]);

    ggml_init_params ip = { ggml_tensor_overhead() * 8192, nullptr, true };
    ggml_context * octx = ggml_init(ip);
    std::vector<std::vector<uint8_t>> bufs;
    size_t in_bytes = 0, out_bytes = 0;
    int n_done = 0;
    const int n_threads = std::max(1u, std::thread::hardware_concurrency());

    for (int a = 5; a < argc; ++a) {
        ggml_context * ctx = nullptr;
        gguf_init_params gp = { true, &ctx };
        gguf_context * g = gguf_init_from_file(argv[a], gp);
        FILE * f = fopen(argv[a], "rb");
        if (!g || !f) {
            fprintf(stderr, "cannot read %s\n", argv[a]);
            return 1;
        }
        const size_t data_off = gguf_get_data_offset(g);
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t = ggml_get_tensor(ctx, name);
            if (!std::regex_search(name, re) || t->type == dst || (!any_src && t->type != src_filter)) {
                continue;
            }
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows     = ggml_nelements(t) / n_per_row;
            if (n_per_row % ggml_blck_size(dst) != 0 || ggml_get_type_traits(t->type)->to_float == nullptr) {
                fprintf(stderr, "skip %s: row of %lld does not fit %s\n", name, (long long) n_per_row, ggml_type_name(dst));
                continue;
            }
            std::vector<uint8_t> raw(ggml_nbytes(t));
            if (fseeko(f, (off_t) (data_off + gguf_get_tensor_offset(g, i)), SEEK_SET) != 0 || fread(raw.data(), 1, raw.size(), f) != raw.size()) {
                fprintf(stderr, "read failed for %s\n", name);
                return 1;
            }
            std::vector<float> imw;
            const bool has_im = column_weights(name, n_per_row, imw);
            if (imat && !has_im) {
                fprintf(stderr, "warning: no imatrix entry for %s\n", name);
            }
            const size_t src_row = ggml_row_size(t->type, n_per_row);
            const size_t dst_row = ggml_row_size(dst, n_per_row);
            bufs.emplace_back(dst_row * nrows);
            uint8_t * q = bufs.back().data();

            // rows split over threads: dequantize to f32, then quantize with ggml's reference quantizer
            std::vector<std::thread> th;
            const int64_t per = (nrows + n_threads - 1) / n_threads;
            for (int k = 0; k < n_threads; ++k) {
                const int64_t r0 = k * per, r1 = std::min(nrows, r0 + per);
                if (r0 >= r1) {
                    break;
                }
                th.emplace_back([&, r0, r1] {
                    std::vector<float> f32((r1 - r0) * n_per_row);
                    ggml_get_type_traits(t->type)->to_float(raw.data() + r0 * src_row, f32.data(), (r1 - r0) * n_per_row);
                    ggml_quantize_chunk(dst, f32.data(), q + r0 * dst_row, 0, r1 - r0, n_per_row, has_im ? imw.data() : nullptr);
                });
            }
            for (auto & x : th) {
                x.join();
            }

            ggml_tensor * ot = ggml_new_tensor(octx, dst, ggml_n_dims(t), t->ne);
            ggml_set_name(ot, name);
            ot->data = q;
            gguf_add_tensor(out, ot);
            in_bytes  += raw.size();
            out_bytes += dst_row * nrows;
            ++n_done;
            fprintf(stderr, "%-48s %s -> %s%s\n", name, ggml_type_name(t->type), ggml_type_name(dst), has_im ? " (imatrix)" : "");
        }
        fclose(f);
        gguf_free(g);
        ggml_free(ctx);
    }
    if (!gguf_write_to_file(out, out_path, false)) {
        fprintf(stderr, "write failed: %s\n", out_path);
        return 1;
    }
    fprintf(stderr, "%d tensors, %.2f GB -> %.2f GB, written to %s\n", n_done, in_bytes / 1e9, out_bytes / 1e9, out_path);
    gguf_free(out);
    ggml_free(octx);
    if (imat) {
        gguf_free(imat);
        ggml_free(ictx);
    }
    return 0;
}
