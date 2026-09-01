#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct llama_ple_source {
    uint64_t tensor_offset;
    uint64_t tensor_bytes;
    uint64_t row_count;
    uint64_t row_elements;
    uint64_t row_bytes;
    uint64_t alignment;
};

struct llama_ple_sector {
    uint64_t offset;
    uint64_t bytes;
};

struct llama_ple_row_span {
    uint64_t row_offset;
    uint64_t first_byte;
    size_t sector_begin;
    size_t sector_count;
};

struct llama_ple_read_wave {
    size_t sector_begin;
    size_t sector_count;
    uint64_t bytes;
};

struct llama_ple_read_plan {
    std::vector<int32_t> rows;
    std::vector<llama_ple_sector> sectors;
    std::vector<llama_ple_row_span> row_spans;
    std::vector<llama_ple_read_wave> waves;
    uint64_t peak_bytes = 0;
};

llama_ple_read_plan llama_ple_plan_rows(
        const llama_ple_source & source,
        const std::vector<int32_t> & rows,
        size_t buffer_bytes);

void llama_ple_dequantize_q8_0_rows(
        const uint8_t * input,
        size_t row_count,
        size_t row_bytes,
        size_t row_elements,
        float * output,
        size_t output_values);

struct llama_ple_pager_stats {
    uint64_t rows = 0;
    uint64_t sectors = 0;
    uint64_t deduplicated_sectors = 0;
    uint64_t bytes = 0;
    uint64_t reads = 0;
    uint64_t failures = 0;
    uint64_t read_us = 0;
    uint64_t max_read_us = 0;
    uint64_t dequant_us = 0;
    uint64_t peak_arena_bytes = 0;
};

class llama_ple_pager {
public:
    static std::unique_ptr<llama_ple_pager> open_from_file_id(
            int file_id,
            const llama_ple_source & source,
            uint32_t io_depth,
            size_t buffer_bytes);

    static std::unique_ptr<llama_ple_pager> open_mmap_from_file_id(
            int file_id,
            const llama_ple_source & source);

    ~llama_ple_pager();

    llama_ple_pager(const llama_ple_pager &) = delete;
    llama_ple_pager & operator=(const llama_ple_pager &) = delete;

    void read_rows(const std::vector<int32_t> & rows, float * output, size_t output_values);
    llama_ple_pager_stats snapshot_stats() const;

private:
    struct impl;
    explicit llama_ple_pager(std::unique_ptr<impl> pimpl);

    std::unique_ptr<impl> pimpl_;
};
