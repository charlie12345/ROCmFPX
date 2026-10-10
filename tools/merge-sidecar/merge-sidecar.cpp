// merge-sidecar: write merged copies of small per-layer weights that share an input into a sidecar GGUF, applied with
// LLAMA_WEIGHT_OVERRIDE=<sidecar> (the loader adds tensors the model file does not have). The original files stay
// untouched; an architecture that finds the merged tensors skips loading the separate ones.
//
// GLM-5.3-Flash KDA layers (glm5-next):
//   blk.N.ssm_fgb_a = rows of ssm_f_a, then ssm_g_a, then ssm_beta  (one GEMM over the layer input)
//   blk.N.ssm_fg_b  = ssm_f_b and ssm_g_b as a batch of 2             (one batched GEMM)
//
// usage: llama-merge-sidecar <out.gguf> <model.gguf> [more splits...]

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct src_tensor {
    ggml_tensor * meta = nullptr;   // shape and type (in its split's context)
    std::string   path;
    size_t        offset = 0;      // absolute file offset of the data
};

static bool read_tensor(const src_tensor & t, std::vector<uint8_t> & out) {
    FILE * f = fopen(t.path.c_str(), "rb");
    if (!f) {
        return false;
    }
    const size_t n = ggml_nbytes(t.meta);
    const size_t old = out.size();
    out.resize(old + n);
    const bool ok = fseeko(f, (off_t) t.offset, SEEK_SET) == 0 && fread(out.data() + old, 1, n, f) == n;
    fclose(f);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <out.gguf> <model.gguf> [more splits...]\n", argv[0]);
        return 1;
    }
    const char * out_path = argv[1];

    std::map<std::string, src_tensor> tensors;
    std::vector<ggml_context *> ctxs;
    std::vector<gguf_context *> ggufs;
    for (int a = 2; a < argc; ++a) {
        ggml_context * ctx = nullptr;
        gguf_init_params gp = { true, &ctx };
        gguf_context * g = gguf_init_from_file(argv[a], gp);
        if (!g) {
            fprintf(stderr, "cannot read %s\n", argv[a]);
            return 1;
        }
        const size_t data_off = gguf_get_data_offset(g);
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            tensors[name] = { ggml_get_tensor(ctx, name), argv[a], data_off + gguf_get_tensor_offset(g, i) };
        }
        ctxs.push_back(ctx);
        ggufs.push_back(g);
    }

    gguf_context * out = gguf_init_empty();
    gguf_set_val_str(out, "general.name", "weight merge sidecar");
    ggml_init_params ip = { ggml_tensor_overhead() * 1024, nullptr, true };
    ggml_context * octx = ggml_init(ip);
    std::vector<std::vector<uint8_t>> bufs;
    bufs.reserve(1024);
    size_t out_bytes = 0;
    int n_layers = 0;

    auto get = [&](const std::string & name) -> const src_tensor * {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    };

    for (int il = 0; il < 1024; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const src_tensor * fa = get(p + "ssm_f_a.weight"), * ga = get(p + "ssm_g_a.weight"), * be = get(p + "ssm_beta.weight");
        const src_tensor * fb = get(p + "ssm_f_b.weight"), * gb = get(p + "ssm_g_b.weight");
        if (!fa || !ga || !be || !fb || !gb) {
            continue;
        }
        // stacked rows need one type and one row width; the 2-batch weight needs identical shapes
        if (fa->meta->type != ga->meta->type || fa->meta->type != be->meta->type ||
                fa->meta->ne[0] != ga->meta->ne[0] || fa->meta->ne[0] != be->meta->ne[0] ||
                ggml_n_dims(fa->meta) != 2 || ggml_n_dims(ga->meta) != 2 || ggml_n_dims(be->meta) != 2 ||
                fa->meta->ne[1] != ga->meta->ne[1] ||
                fb->meta->type != gb->meta->type || !ggml_are_same_shape(fb->meta, gb->meta) || ggml_n_dims(fb->meta) != 2) {
            fprintf(stderr, "layer %d: shapes or types do not allow merging, skipped\n", il);
            continue;
        }

        bufs.emplace_back();
        std::vector<uint8_t> & a = bufs.back();
        if (!read_tensor(*fa, a) || !read_tensor(*ga, a) || !read_tensor(*be, a)) {
            fprintf(stderr, "read failed in layer %d\n", il);
            return 1;
        }
        const int64_t ne_a[2] = { fa->meta->ne[0], fa->meta->ne[1] + ga->meta->ne[1] + be->meta->ne[1] };
        ggml_tensor * ta = ggml_new_tensor(octx, fa->meta->type, 2, ne_a);
        ggml_set_name(ta, (p + "ssm_fgb_a.weight").c_str());
        ta->data = a.data();
        gguf_add_tensor(out, ta);

        bufs.emplace_back();
        std::vector<uint8_t> & b = bufs.back();
        if (!read_tensor(*fb, b) || !read_tensor(*gb, b)) {
            fprintf(stderr, "read failed in layer %d\n", il);
            return 1;
        }
        const int64_t ne_b[3] = { fb->meta->ne[0], fb->meta->ne[1], 2 };
        ggml_tensor * tb = ggml_new_tensor(octx, fb->meta->type, 3, ne_b);
        ggml_set_name(tb, (p + "ssm_fg_b.weight").c_str());
        tb->data = b.data();
        gguf_add_tensor(out, tb);

        out_bytes += a.size() + b.size();
        ++n_layers;
        fprintf(stderr, "layer %3d: ssm_fgb_a %s [%lld, %lld], ssm_fg_b %s [%lld, %lld, 2]\n", il,
            ggml_type_name(ta->type), (long long) ne_a[0], (long long) ne_a[1],
            ggml_type_name(tb->type), (long long) ne_b[0], (long long) ne_b[1]);
    }
    if (n_layers == 0) {
        fprintf(stderr, "no mergeable layers found\n");
        return 1;
    }
    if (!gguf_write_to_file(out, out_path, false)) {
        fprintf(stderr, "write failed: %s\n", out_path);
        return 1;
    }
    fprintf(stderr, "%d layers, %.1f MB, written to %s\n", n_layers, out_bytes / 1e6, out_path);
    gguf_free(out);
    ggml_free(octx);
    for (auto * g : ggufs) {
        gguf_free(g);
    }
    for (auto * c : ctxs) {
        ggml_free(c);
    }
    return 0;
}
