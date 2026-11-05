#include "common.cuh"
#include "expand.cuh"
#include <cstdio>
#include <cstdlib>
#include <string>

struct expand_layout {
    size_t block_row, score_row, cell_row, tail_row, output_row, cast_block_row, cast_cell_row, expanded_row;
    int n_blocks;
};

__global__ __launch_bounds__(256) void qsa_expand_complete_blocks_512(
        const char * blocks, const char * scores, const char * cells, const char * tail,
        char * output, char * cast_blocks, char * cast_cells, char * expanded, expand_layout layout) {
    const int query = blockIdx.x;
    const int lane = threadIdx.x;
    __shared__ __align__(16) int ordered[512];
    const int * selected = reinterpret_cast<const int *>(blocks + query*layout.block_row);
    for (int i = lane; i < 512; i += 256) {
        ordered[i] = selected[i];
        reinterpret_cast<float *>(cast_blocks + query*layout.cast_block_row)[i] = float(selected[i]);
    }
    __syncthreads();
    for (int size = 2; size <= 512; size *= 2) {
        for (int distance = size/2; distance; distance /= 2) {
            for (int i = lane; i < 512; i += 256) {
                const int other = i ^ distance;
                if (other > i) {
                    const int a = ordered[i], b = ordered[other];
                    if ((i & size) ? a < b : a > b) { ordered[i] = b; ordered[other] = a; }
                }
            }
            __syncthreads();
        }
    }
    const float * score = reinterpret_cast<const float *>(scores + query*layout.score_row);
    const int * remainder = reinterpret_cast<const int *>(tail + query*layout.tail_row);
    int * out = reinterpret_cast<int *>(output + query*layout.output_row);
    for (int i = lane; i < 2051; i += 256) {
        if (i >= 2048) { out[i] = remainder[i-2048]; continue; }
        const int block = ordered[i/4];
        if (block < 0 || block >= layout.n_blocks) { out[i] = -1; continue; }
        const int cell = reinterpret_cast<const int *>(cells + block*layout.cell_row)[i%4];
        const float valid = __fadd_rn(score[block], 1.0f) > 0.0f ? 1.0f : 0.0f;
        const float value = float(cell);
        reinterpret_cast<float *>(cast_cells + query*layout.cast_cell_row)[i] = value;
        const float shifted = __fadd_rn(value, 1.0f);
        const int result = int(__fadd_rn(__fmul_rn(shifted, valid), -1.0f));
        reinterpret_cast<int *>(expanded + query*layout.expanded_row)[i] = result;
        out[i] = result;
    }
}

void ggml_cuda_op_qsa_expand(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_expand_args & a) {
    expand_layout layout{a.blocks->nb[1], a.scores->nb[1], a.cells->nb[1], a.tail->nb[1], a.output->nb[1], a.cast_blocks->nb[1], a.cast_cells->nb[2], a.expanded->nb[2], int(a.scores->ne[0])};
    const ggml_cuda_kernel_launch_params launch(dim3(a.output->ne[1]), dim3(256), 0, ctx.stream());
    ggml_cuda_kernel_launch(qsa_expand_complete_blocks_512, launch,
        (const char *)a.blocks->data, (const char *)a.scores->data, (const char *)a.cells->data,
        (const char *)a.tail->data, (char *)a.output->data, (char *)a.cast_blocks->data,
        (char *)a.cast_cells->data, (char *)a.expanded->data, layout);
    CUDA_CHECK(cudaGetLastError());
}
