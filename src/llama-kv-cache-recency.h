#pragma once

#include "llama-batch.h"
#include "llama-memory.h"
#include "llama-kv-recency-plan.h"

#include <unordered_map>

struct llama_hparams;
struct llama_model;

class LLAMA_API llama_kv_cache_recency : public llama_memory_i {
public:
    llama_kv_cache_recency(llama_memory_ptr delegate, const llama_kv_recency_plan & plan);
    ~llama_kv_cache_recency() override;

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;
    bool get_can_shift() const override;
    void clear(bool data) override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) override;

    const llama_kv_recency_plan & get_plan() const { return plan; }
    static llama_memory_t unwrap(llama_memory_t memory);

private:
    llama_memory_ptr delegate;
    const llama_kv_recency_plan plan;
};

struct llama_kv_recency_row {
    std::vector<uint8_t> k;
    std::vector<uint8_t> v;
};

class llama_kv_cells;

class LLAMA_API llama_kv_recency_store {
public:
    enum class mode { fixed_snapshot, committed_aging };
    struct migration_stats {
        uint32_t cold_pages = 0;
        uint32_t hot_pages = 0;
        size_t migrated_source_bytes = 0;
        size_t resident_payload_bytes = 0;
        size_t peak_workspace_bytes = 0;
    };
    explicit llama_kv_recency_store(const llama_kv_recency_plan_params & params, mode layout = mode::fixed_snapshot);
    ~llama_kv_recency_store();

    // Committed mode throws std::runtime_error, without changes, when a new page finds no free Q8 ring slot.
    void load_q8_row(uint32_t cell, llama_pos position, const std::vector<llama_seq_id> & sequences,
                     const std::vector<llama_kv_recency_row> & rows);
    std::vector<uint8_t> read_row(uint32_t layer, bool key, uint32_t cell) const;
    // Committed mode: commit n <= migrate_tokens consecutive cells whose Q8 rows the caller writes before any reader runs.
    // New pages take ring slots and slabs first; throws std::runtime_error, without logical changes, when they cannot.
    // After a full-ring failure one migration call frees enough slots. Returns the first cell.
    uint32_t append(llama_pos position, llama_seq_id seq, uint32_t n);
    // Q8 slab and row of a resident Q8 cell.
    std::pair<uint32_t, uint32_t> get_q8_row(uint32_t cell) const;
    // Removes cells >= cell. Returns false, without changes, when that would split a Q4 page.
    bool truncate(uint32_t cell);
    bool can_truncate(uint32_t cell) const;
    uint32_t get_n_cells() const;
    const llama_kv_recency_plan & get_plan() const;
    void allocate_slab(ggml_type type, uint32_t slab);
    // Slab positions of a region. An unallocated slab has no tensors.
    uint32_t get_slab_count(ggml_type type) const;
    ggml_tensor * get_tensor(uint32_t layer, bool key, ggml_type type, uint32_t slab = 0) const;
    // Logical cell held by a row of a Q4_0 or Q8_0 slab, or -1 when the row is unused.
    int32_t get_row_cell(ggml_type type, uint32_t slab, uint32_t row) const;
    const llama_kv_cells & get_cells() const;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;
    // Caller excludes concurrent submissions/readers until return. All reading backends are synchronized first.
    uint32_t migrate_oldest_pages(const std::vector<ggml_backend_t> & readers, uint32_t max_pages);
    migration_stats get_migration_stats() const;

    // Versioned committed-mode state: magic, version and session geometry, then rows in logical order
    // (q4_0 pages, then q8_0 pages). A sequence it does not hold is written as an empty section.
    void state_write(llama_io_write_i & io, llama_seq_id seq_id) const;
    struct state_header {
        uint32_t n_cells = 0;
        uint32_t cold_pages = 0;
        llama_pos first_position = -1;
        llama_seq_id sequence = -1;
        std::vector<uint32_t> slots; // ring slot of each q8_0 page, oldest first
    };
    // Reads a header and checks it against this session's geometry. Throws, without changes, when it does not fit.
    state_header state_read_header(llama_io_read_i & io) const;
    // Replaces the contents with the header's cells and reads their rows; seq_id < 0 keeps the saved sequence.
    void state_read_rows(llama_io_read_i & io, const state_header & header, llama_seq_id seq_id);

private:
    struct impl;
    std::unique_ptr<impl> data;
};

// Live elastic recency memory for one dense sequence: committed KV enters a Q8 ring and ages to Q4 by position.
class LLAMA_API llama_kv_cache_recency_live : public llama_memory_i {
public:
    llama_kv_cache_recency_live(const llama_model & model, const std::vector<uint32_t> & layer_ids,
                                const llama_kv_recency_plan_params & params, std::vector<ggml_backend_t> readers);
    ~llama_kv_cache_recency_live() override;

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;
    bool get_can_shift() const override;
    void clear(bool data) override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) override;

    llama_kv_recency_store & get_store() { return store; }
    const llama_kv_recency_store & get_store() const { return store; }
    const llama_hparams & get_hparams() const { return hparams; }
    const std::vector<ggml_backend_t> & get_readers() const { return readers; }
    // store layer of a model layer
    uint32_t get_layer(int32_t il) const { return layer_map.at(il); }
    uint32_t get_n_rot_k() const { return n_rot_k; }
    uint32_t get_n_rot_v() const { return n_rot_v; }
    const std::vector<float> & get_rot(uint32_t n) const { return hadamard.at(n); }

private:
    const llama_hparams & hparams;
    llama_kv_recency_store store;
    std::unordered_map<int32_t, uint32_t> layer_map;
    std::vector<ggml_backend_t> readers;
    uint32_t n_rot_k = 0;
    uint32_t n_rot_v = 0;
    std::map<uint32_t, std::vector<float>> hadamard;

    // range of the stored sequence, or false when it does not hold seq_id
    bool seq_range(llama_seq_id seq_id, llama_pos & first, llama_pos & last) const;
    void synchronize() const;
};

class LLAMA_API llama_kv_cache_recency_context : public llama_memory_context_i {
public:
    struct region {
        ggml_type type;
        uint32_t slab;
    };

    explicit llama_kv_cache_recency_context(llama_memory_status status);
    // full context for graph reservation
    explicit llama_kv_cache_recency_context(llama_kv_cache_recency_live * mem);
    llama_kv_cache_recency_context(llama_kv_cache_recency_live * mem, std::vector<llama_ubatch> ubatches);
    ~llama_kv_cache_recency_context() override;

    bool next() override;
    bool apply() override;
    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    // allocated Q4 and Q8 slabs; each is one homogeneous attention region
    std::vector<region> get_regions() const;
    uint32_t get_rows(const region & r) const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il, const region & r) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, const region & r) const;

    // store the ubatch K/V rows into its Q8 slab; idxs are rows of that slab
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il) const;

    uint32_t get_n_rot_k() const;
    uint32_t get_n_rot_v() const;

    void set_input_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_kq_mask(ggml_tensor * dst, const region & r, const llama_ubatch * ubatch) const;
    void set_input_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;
    llama_kv_cache_recency_live * mem = nullptr;
    std::vector<llama_ubatch> ubatches;
    size_t i_cur = 0;
    uint32_t first_cell = 0;
    uint32_t dst_slab = 0;
};
