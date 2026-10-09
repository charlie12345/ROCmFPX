#include "llama-kv-cache-recency.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

llama_kv_cache_recency::llama_kv_cache_recency(llama_memory_ptr delegate, const llama_kv_recency_plan & plan) :
    delegate(std::move(delegate)), plan(plan) {
    if (!this->delegate) {
        throw std::invalid_argument("recency wrapper requires a memory delegate");
    }
}

llama_kv_cache_recency::~llama_kv_cache_recency() = default;

llama_memory_context_ptr llama_kv_cache_recency::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    return delegate->init_batch(balloc, n_ubatch, embd_all);
}

llama_memory_context_ptr llama_kv_cache_recency::init_full() {
    return delegate->init_full();
}

llama_memory_context_ptr llama_kv_cache_recency::init_update(llama_context * lctx, bool optimize) {
    return delegate->init_update(lctx, optimize);
}

bool llama_kv_cache_recency::get_can_shift() const { return delegate->get_can_shift(); }
void llama_kv_cache_recency::clear(bool data) { delegate->clear(data); }
bool llama_kv_cache_recency::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) { return delegate->seq_rm(seq_id, p0, p1); }
void llama_kv_cache_recency::seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) { delegate->seq_cp(src, dst, p0, p1); }
void llama_kv_cache_recency::seq_keep(llama_seq_id seq_id) { delegate->seq_keep(seq_id); }
void llama_kv_cache_recency::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) { delegate->seq_add(seq_id, p0, p1, shift); }
void llama_kv_cache_recency::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) { delegate->seq_div(seq_id, p0, p1, d); }
llama_pos llama_kv_cache_recency::seq_pos_min(llama_seq_id seq_id) const { return delegate->seq_pos_min(seq_id); }
llama_pos llama_kv_cache_recency::seq_pos_max(llama_seq_id seq_id) const { return delegate->seq_pos_max(seq_id); }
std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_recency::memory_breakdown() const { return delegate->memory_breakdown(); }
void llama_kv_cache_recency::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const { delegate->state_write(io, seq_id, flags); }
void llama_kv_cache_recency::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) { delegate->state_read(io, seq_id, flags); }

llama_memory_t llama_kv_cache_recency::unwrap(llama_memory_t memory) {
    while (auto * recency = dynamic_cast<llama_kv_cache_recency *>(memory)) {
        memory = recency->delegate.get();
    }
    return memory;
}

struct llama_kv_recency_store::impl {
    struct slab_t {
        std::vector<ggml_tensor *> k;
        std::vector<ggml_tensor *> v;
        std::vector<ggml_context_ptr> contexts;
        std::vector<std::pair<ggml_backend_buffer_type_t, ggml_backend_buffer_ptr>> buffers;
    };
    llama_kv_recency_plan_params params;
    llama_kv_recency_plan plan;
    mode layout = mode::fixed_snapshot;
    uint32_t next_cell = 0;
    uint32_t cold_pages = 0;
    llama_pos first_position = -1;
    llama_seq_id sequence = -1;
    size_t peak_workspace_bytes = 0;
    std::vector<int32_t> page_slot; // logical page -> Q8 ring slot, -1 when not in Q8
    std::vector<int32_t> slot_page; // Q8 ring slot -> logical page, -1 when free
    uint32_t region_rows[2] = {};   // region 0 is Q4 cold, region 1 is Q8 hot
    uint32_t slab_rows[2] = {};
    std::vector<std::unique_ptr<slab_t>> slabs[2];
    llama_kv_cells cells;

    std::pair<uint32_t, uint32_t> locate(uint32_t cell) const {
        if (cell >= plan.logical_ctx) {
            throw std::out_of_range("KV recency logical cell exceeds context");
        }
        if (layout == mode::committed_aging) {
            const uint32_t page = cell/plan.page_tokens;
            if (page < cold_pages) {
                return std::make_pair(0u, cell);
            }
            int32_t slot = page_slot[page];
            if (slot < 0) {
                // First row of a new page takes a free slot; occupied slots are never overwritten.
                const auto free = std::find(slot_page.begin(), slot_page.end(), -1);
                if (free == slot_page.end()) {
                    throw std::runtime_error("KV recency Q8 ring is full; migrate eligible pages first");
                }
                slot = int32_t(free - slot_page.begin());
            }
            return std::make_pair(1u, uint32_t(slot)*plan.page_tokens + cell%plan.page_tokens);
        }
        return cell < plan.cold_tokens ? std::make_pair(0u, cell) : std::make_pair(1u, cell - plan.cold_tokens);
    }

    slab_t & allocate(uint32_t region, uint32_t index) {
        auto & slab = slabs[region].at(index);
        if (slab) {
            return *slab;
        }
        const uint32_t tokens = std::min(slab_rows[region], region_rows[region] - index*slab_rows[region]);
        const ggml_type type = region == 0 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        std::unique_ptr<slab_t> result(new slab_t);
        result->k.resize(params.layers.size());
        result->v.resize(params.layers.size());
        std::map<ggml_backend_buffer_type_t, ggml_context_ptr> contexts;
        for (size_t i = 0; i < params.layers.size(); ++i) {
            const auto & layer = params.layers[i];
            auto & ctx = contexts[layer.buft];
            if (!ctx) {
                const ggml_init_params init = {2*params.layers.size()*ggml_tensor_overhead(), nullptr, true};
                ctx.reset(ggml_init(init));
                if (!ctx) {
                    throw std::runtime_error("failed to allocate recency tensor metadata");
                }
            }
            result->k[i] = ggml_new_tensor_3d(ctx.get(), type, layer.k_width, tokens, 1);
            result->v[i] = layer.v_width ? ggml_new_tensor_3d(ctx.get(), type, layer.v_width, tokens, 1) : nullptr;
        }
        for (auto & entry : contexts) {
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(entry.second.get(), entry.first));
            if (!buffer) {
                throw std::runtime_error("failed to allocate KV recency slab");
            }
            ggml_backend_buffer_clear(buffer.get(), 0);
            result->buffers.emplace_back(entry.first, std::move(buffer));
            result->contexts.push_back(std::move(entry.second));
        }
        slab = std::move(result);
        return *slab;
    }

    // session geometry a saved state must match
    std::vector<uint32_t> geometry() const {
        std::vector<uint32_t> result = {plan.logical_ctx, plan.page_tokens, plan.migrate_tokens, plan.hot_tokens, plan.cold_tokens,
                                        plan.ring_tokens, slab_rows[0], slab_rows[1], uint32_t(params.layers.size())};
        for (const auto & layer : params.layers) {
            result.push_back(uint32_t(layer.k_width));
            result.push_back(uint32_t(layer.v_width));
        }
        return result;
    }

    // Calls f(tensor, offset, size) for every resident row in logical order: per layer K then V, q4_0 pages, then q8_0 pages.
    template <typename F> void for_each_row(F && f) const {
        const uint32_t page_tokens = plan.page_tokens;
        const auto rows = [&](uint32_t region, uint32_t layer, bool key, uint32_t row, uint32_t n) {
            while (n) {
                size_t offset = 0;
                ggml_tensor * t = tensor(region, layer, key, row, offset);
                const uint32_t count = std::min(n, slab_rows[region] - row % slab_rows[region]);
                f(t, offset, size_t(count)*t->nb[1]);
                row += count;
                n -= count;
            }
        };
        for (uint32_t layer = 0; layer < params.layers.size(); ++layer) {
            for (bool key : {true, false}) {
                if (!key && !params.layers[layer].v_width) {
                    continue;
                }
                rows(0, layer, key, 0, cold_pages*page_tokens);
                for (uint32_t page = cold_pages; page*page_tokens < next_cell; ++page) {
                    rows(1, layer, key, uint32_t(page_slot[page])*page_tokens, std::min(page_tokens, next_cell - page*page_tokens));
                }
            }
        }
    }

    ggml_tensor * tensor(uint32_t region, uint32_t layer, bool key, uint32_t row, size_t & offset) const {
        const auto & slab = slabs[region].at(row/slab_rows[region]);
        ggml_tensor * result = key ? slab->k.at(layer) : slab->v.at(layer);
        if (!result) {
            throw std::invalid_argument("KV recency layer has no V rows");
        }
        offset = size_t(row % slab_rows[region])*result->nb[1];
        return result;
    }
};

static uint32_t recency_region(ggml_type type) {
    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q8_0) {
        throw std::invalid_argument("KV recency region must be Q4_0 or Q8_0");
    }
    return type == GGML_TYPE_Q4_0 ? 0 : 1;
}

llama_kv_recency_store::llama_kv_recency_store(const llama_kv_recency_plan_params & params, mode layout) : data(new impl) {
    data->params = params;
    data->plan = llama_kv_recency_plan_create(params);
    data->layout = layout;
    if (params.streams != 1) {
        throw std::invalid_argument("fixed KV recency store requires one unified stream");
    }
    data->cells.resize(params.logical_ctx);
    data->region_rows[0] = data->plan.cold_tokens;
    data->region_rows[1] = layout == mode::committed_aging ? data->plan.ring_tokens : data->plan.hot_tokens;
    for (uint32_t region = 0; region < 2; ++region) {
        // Committed slabs are allocated on first use; the fixed snapshot keeps one eager slab per region.
        const uint32_t rows = data->region_rows[region];
        data->slab_rows[region] = layout == mode::fixed_snapshot ? rows :
            (region == 1 && params.ring_one_slab ? std::max(rows, 1u) : data->plan.slab_tokens);
        data->slabs[region].resize(rows ? (rows + data->slab_rows[region] - 1)/data->slab_rows[region] : 0);
        if (layout == mode::fixed_snapshot && rows) {
            data->allocate(region, 0);
        }
    }
    if (layout == mode::committed_aging) {
        data->page_slot.assign(data->plan.physical_tokens/data->plan.page_tokens, -1);
        data->slot_page.assign(data->plan.ring_tokens/data->plan.page_tokens, -1);
    }
}

llama_kv_recency_store::~llama_kv_recency_store() = default;

void llama_kv_recency_store::load_q8_row(uint32_t cell, llama_pos position,
                                       const std::vector<llama_seq_id> & sequences,
                                       const std::vector<llama_kv_recency_row> & rows) {
    const auto location = data->locate(cell);
    if (!data->cells.is_empty(cell) || position < 0 || sequences.empty() || rows.size() != data->params.layers.size()) {
        throw std::invalid_argument("invalid KV recency snapshot row or live overwrite");
    }
    if (data->layout == mode::committed_aging &&
            (cell != data->next_cell || sequences.size() != 1 ||
             (cell && (int64_t(position) != int64_t(data->first_position) + cell || sequences[0] != data->sequence)))) {
        throw std::invalid_argument("committed recency prototype requires append-only positions and one sequence");
    }
    for (const auto seq : sequences) {
        if (seq < 0 || seq >= LLAMA_MAX_SEQ) {
            throw std::invalid_argument("invalid KV recency sequence ID");
        }
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & layer = data->params.layers[i];
        if (rows[i].k.size() != ggml_row_size(GGML_TYPE_Q8_0, layer.k_width) ||
            rows[i].v.size() != ggml_row_size(GGML_TYPE_Q8_0, layer.v_width)) {
            throw std::invalid_argument("KV recency snapshot row byte count mismatch");
        }
    }
    data->allocate(location.first, location.second/data->slab_rows[location.first]);
    // Readers only see the cell after all synchronous tensor writes finish.
    for (size_t i = 0; i < rows.size(); ++i) {
        for (bool key : {true, false}) {
            if (!key && !data->params.layers[i].v_width) {
                continue;
            }
            size_t offset = 0;
            ggml_tensor * tensor = data->tensor(location.first, i, key, location.second, offset);
            const auto & source = key ? rows[i].k : rows[i].v;
            const size_t bytes = ggml_row_size(tensor->type, tensor->ne[0]);
            if (tensor->type == GGML_TYPE_Q8_0) {
                ggml_backend_tensor_set(tensor, source.data(), offset, bytes);
            } else {
                std::vector<float> values(tensor->ne[0]);
                std::vector<uint8_t> packed(bytes);
                ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(source.data(), values.data(), tensor->ne[0]);
                ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), packed.data(), 0, 1, tensor->ne[0], nullptr);
                ggml_backend_tensor_set(tensor, packed.data(), offset, bytes);
            }
        }
    }
    data->cells.pos_set(cell, position);
    for (const auto seq : sequences) {
        data->cells.seq_add(cell, seq);
    }
    if (data->layout == mode::committed_aging) {
        if (cell == 0) { data->first_position = position; data->sequence = sequences[0]; }
        const uint32_t page = cell/data->plan.page_tokens;
        if (location.first == 1 && data->page_slot[page] < 0) {
            const uint32_t slot = location.second/data->plan.page_tokens;
            data->page_slot[page] = int32_t(slot);
            data->slot_page[slot] = int32_t(page);
        }
        ++data->next_cell;
    }
}

std::vector<uint8_t> llama_kv_recency_store::read_row(uint32_t layer, bool key, uint32_t cell) const {
    if (cell >= data->plan.logical_ctx || data->cells.is_empty(cell)) {
        throw std::invalid_argument("KV recency cell is not resident");
    }
    const auto location = data->locate(cell);
    size_t offset = 0;
    ggml_tensor * tensor = data->tensor(location.first, layer, key, location.second, offset);
    std::vector<uint8_t> result(ggml_row_size(tensor->type, tensor->ne[0]));
    ggml_backend_tensor_get(tensor, result.data(), offset, result.size());
    return result;
}

uint32_t llama_kv_recency_store::append(llama_pos position, llama_seq_id seq, uint32_t n) {
    auto & d = *data;
    // One migration call can free enough slots only for appends up to one migration batch.
    if (d.layout != mode::committed_aging || !n || n > d.plan.migrate_tokens || n > d.plan.logical_ctx - d.next_cell || position < 0 ||
            seq < 0 || seq >= LLAMA_MAX_SEQ || (d.next_cell && (int64_t(position) != int64_t(d.first_position) + d.next_cell || seq != d.sequence))) {
        throw std::invalid_argument("committed recency append requires consecutive positions of one sequence within context");
    }
    const uint32_t page_tokens = d.plan.page_tokens;
    std::vector<uint32_t> pages;
    for (uint32_t page = d.next_cell/page_tokens; page <= (d.next_cell + n - 1)/page_tokens; ++page) {
        if (d.page_slot[page] < 0) {
            pages.push_back(page);
        }
    }
    std::vector<uint32_t> slots;
    for (uint32_t slot = 0; slot < d.slot_page.size() && slots.size() < pages.size(); ++slot) {
        if (d.slot_page[slot] < 0) {
            slots.push_back(slot);
        }
    }
    if (slots.size() < pages.size()) {
        throw std::runtime_error("KV recency Q8 ring is full; migrate eligible pages first");
    }
    for (const uint32_t slot : slots) {
        d.allocate(1, slot*page_tokens/d.slab_rows[1]);
    }
    for (size_t i = 0; i < pages.size(); ++i) {
        d.page_slot[pages[i]] = int32_t(slots[i]);
        d.slot_page[slots[i]] = int32_t(pages[i]);
    }
    const uint32_t first = d.next_cell;
    for (uint32_t i = 0; i < n; ++i) {
        d.cells.pos_set(first + i, position + llama_pos(i));
        d.cells.seq_add(first + i, seq);
    }
    if (!first) {
        d.first_position = position;
        d.sequence = seq;
    }
    d.next_cell += n;
    return first;
}

std::pair<uint32_t, uint32_t> llama_kv_recency_store::get_q8_row(uint32_t cell) const {
    if (cell >= data->plan.logical_ctx || data->cells.is_empty(cell)) {
        throw std::invalid_argument("KV recency cell is not resident");
    }
    const auto location = data->locate(cell);
    if (location.first != 1) {
        throw std::invalid_argument("KV recency cell is not in the Q8 ring");
    }
    return std::make_pair(location.second/data->slab_rows[1], location.second % data->slab_rows[1]);
}

bool llama_kv_recency_store::can_truncate(uint32_t cell) const {
    const uint32_t page_tokens = data->plan.page_tokens;
    return cell >= data->next_cell || cell/page_tokens >= data->cold_pages || cell % page_tokens == 0;
}

bool llama_kv_recency_store::truncate(uint32_t cell) {
    auto & d = *data;
    if (d.layout != mode::committed_aging) {
        throw std::invalid_argument("truncation requires committed recency storage");
    }
    const uint32_t page_tokens = d.plan.page_tokens;
    if (cell >= d.next_cell) {
        return true;
    }
    if (!can_truncate(cell)) {
        return false;
    }
    for (uint32_t page = std::max(cell/page_tokens, d.cold_pages); page*page_tokens < d.next_cell; ++page) {
        const uint32_t slot = uint32_t(d.page_slot[page]);
        const uint32_t keep = page*page_tokens < cell ? cell - page*page_tokens : 0;
        const auto & slab = *d.slabs[1].at(slot*page_tokens/d.slab_rows[1]);
        const size_t row = slot*page_tokens % d.slab_rows[1] + keep;
        for (size_t i = 0; i < d.params.layers.size(); ++i) {
            for (auto * tensor : {slab.k[i], slab.v[i]}) {
                if (tensor) {
                    ggml_backend_tensor_memset(tensor, 0, row*tensor->nb[1], size_t(page_tokens - keep)*tensor->nb[1]);
                }
            }
        }
        if (!keep) {
            d.page_slot[page] = -1;
            d.slot_page[slot] = -1;
        }
    }
    d.cold_pages = std::min(d.cold_pages, cell/page_tokens);
    for (uint32_t i = cell; i < d.next_cell; ++i) {
        d.cells.rm(i);
    }
    d.next_cell = cell;
    if (!cell) {
        d.first_position = -1;
        d.sequence = -1;
    }
    return true;
}

const llama_kv_recency_plan & llama_kv_recency_store::get_plan() const {
    return data->plan;
}

uint32_t llama_kv_recency_store::get_n_cells() const {
    return data->next_cell;
}

void llama_kv_recency_store::allocate_slab(ggml_type type, uint32_t slab) {
    data->allocate(recency_region(type), slab);
}

uint32_t llama_kv_recency_store::get_slab_count(ggml_type type) const {
    return uint32_t(data->slabs[recency_region(type)].size());
}

ggml_tensor * llama_kv_recency_store::get_tensor(uint32_t layer, bool key, ggml_type type, uint32_t slab) const {
    const auto & slabs = data->slabs[recency_region(type)];
    if (slab >= slabs.size() || !slabs[slab]) {
        return nullptr;
    }
    return key ? slabs[slab]->k.at(layer) : slabs[slab]->v.at(layer);
}

int32_t llama_kv_recency_store::get_row_cell(ggml_type type, uint32_t slab, uint32_t row) const {
    auto * tensor = get_tensor(0, true, type, slab);
    if (!tensor || row >= tensor->ne[1]) {
        throw std::out_of_range("KV recency physical row exceeds allocated slab");
    }
    uint32_t cell = slab*data->slab_rows[recency_region(type)] + row;
    if (type == GGML_TYPE_Q4_0 && data->layout == mode::committed_aging && cell/data->plan.page_tokens >= data->cold_pages) {
        return -1;
    }
    if (type == GGML_TYPE_Q8_0 && data->layout == mode::fixed_snapshot) {
        cell += data->plan.cold_tokens;
    }
    if (type == GGML_TYPE_Q8_0 && data->layout == mode::committed_aging) {
        const int32_t page = data->slot_page[cell/data->plan.page_tokens];
        if (page < 0) {
            return -1;
        }
        cell = uint32_t(page)*data->plan.page_tokens + cell%data->plan.page_tokens;
    }
    return cell < data->plan.logical_ctx && !data->cells.is_empty(cell) ? int32_t(cell) : -1;
}

const llama_kv_cells & llama_kv_recency_store::get_cells() const {
    return data->cells;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_recency_store::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & region : data->slabs) {
        for (const auto & slab : region) {
            if (!slab) {
                continue;
            }
            for (const auto & entry : slab->buffers) {
                result[entry.first] += ggml_backend_buffer_get_size(entry.second.get());
            }
        }
    }
    return result;
}

uint32_t llama_kv_recency_store::migrate_oldest_pages(const std::vector<ggml_backend_t> & readers, uint32_t max_pages) {
    if (data->layout != mode::committed_aging) {
        throw std::invalid_argument("page aging requires committed recency storage");
    }
    std::map<ggml_backend_buffer_type_t, ggml_backend_t> backends;
    for (const auto & layer : data->params.layers) {
        for (auto * backend : readers) {
            const auto * owner = ggml_backend_buft_get_device(layer.buft);
            if (backend && (owner ? ggml_backend_get_device(backend) == owner :
                    ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) &&
                    ggml_backend_dev_supports_buft(ggml_backend_get_device(backend), layer.buft)) {
                backends[layer.buft] = backend;
                break;
            }
        }
        if (!backends.count(layer.buft)) {
            throw std::invalid_argument("migration requires a reading backend for every KV device");
        }
    }
    // Synchronous prototype: caller holds its graph-submission lock until this method returns.
    for (auto * backend : readers) {
        if (!backend) { throw std::invalid_argument("null KV reading backend"); }
        ggml_backend_synchronize(backend);
    }
    const uint32_t page_tokens = data->plan.page_tokens;
    const uint32_t limit = std::min(max_pages, data->plan.migrate_tokens/page_tokens);
    uint32_t migrated = 0;
    while (migrated < limit && data->cold_pages < data->plan.cold_tokens/page_tokens) {
        const uint32_t begin = data->cold_pages*page_tokens;
        // Full pages only; a page intersecting the newest hot_tokens remains entirely Q8.
        if (data->next_cell < begin + page_tokens ||
                int64_t(data->cells.pos_get(begin + page_tokens - 1)) >=
                    int64_t(data->first_position) + data->next_cell - data->plan.hot_tokens) {
            break;
        }
        const uint32_t slot_row = uint32_t(data->page_slot[data->cold_pages])*page_tokens;
        const auto & q8 = *data->slabs[1].at(slot_row/data->slab_rows[1]);
        const auto & q4 = data->allocate(0, begin/data->slab_rows[0]);
        const size_t q8_row = slot_row % data->slab_rows[1];
        const size_t q4_row = begin % data->slab_rows[0];
        struct job {
            ggml_backend_t backend;
            ggml_context_ptr ctx;
            ggml_backend_buffer_ptr workspace;
            ggml_cgraph * graph;
        };
        std::vector<job> jobs;
        size_t workspace_bytes = 0;
        // Allocate/validate every bounded destination graph before changing any mapping.
        for (const auto & group : backends) {
            const size_t nodes = 16*data->params.layers.size();
            ggml_context_ptr ctx(ggml_init({nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
            if (!ctx) { throw std::runtime_error("failed to allocate migration graph metadata"); }
            auto * graph = ggml_new_graph_custom(ctx.get(), nodes, false);
            for (size_t i = 0; i < data->params.layers.size(); ++i) {
                if (data->params.layers[i].buft != group.first) { continue; }
                for (bool key : {true, false}) {
                    auto * source = key ? q8.k[i] : q8.v[i];
                    auto * destination = key ? q4.k[i] : q4.v[i];
                    if (!source) {
                        continue;
                    }
                    auto * from = ggml_view_2d(ctx.get(), source, source->ne[0], page_tokens, source->nb[1], q8_row*source->nb[1]);
                    auto * to = ggml_view_2d(ctx.get(), destination, destination->ne[0], page_tokens, destination->nb[1], q4_row*destination->nb[1]);
                    auto * f32 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, source->ne[0], page_tokens);
                    auto * unpack = ggml_cpy(ctx.get(), from, f32);
                    auto * pack = ggml_cpy(ctx.get(), unpack, to);
                    ggml_build_forward_expand(graph, pack);
                }
            }
            for (int node = 0; node < ggml_graph_n_nodes(graph); ++node) {
                if (!ggml_backend_supports_op(group.second, ggml_graph_node(graph, node))) {
                    throw std::runtime_error("backend does not support device Q8/F32/Q4 page conversion");
                }
            }
            ggml_backend_buffer_ptr workspace(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), group.first));
            if (!workspace) { throw std::runtime_error("failed to allocate bounded KV migration workspace"); }
            workspace_bytes += ggml_backend_buffer_get_size(workspace.get());
            jobs.push_back({group.second, std::move(ctx), std::move(workspace), graph});
        }
        data->peak_workspace_bytes = std::max(data->peak_workspace_bytes, workspace_bytes);
        for (auto & job : jobs) {
            if (ggml_backend_graph_compute(job.backend, job.graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("KV page conversion failed before mapping publication");
            }
            ggml_backend_synchronize(job.backend);
        }
        // All Q4 K/V layers are complete. Readers were drained and cannot resubmit until return.
        data->slot_page[slot_row/page_tokens] = -1;
        data->page_slot[data->cold_pages] = -1;
        ++data->cold_pages;
        for (size_t i = 0; i < data->params.layers.size(); ++i) {
            for (auto * source : {q8.k[i], q8.v[i]}) {
                if (!source) {
                    continue;
                }
                ggml_backend_tensor_memset(source, 0, q8_row*source->nb[1], size_t(page_tokens)*source->nb[1]);
            }
        }
        ++migrated;
    }
    return migrated;
}

llama_kv_recency_store::migration_stats llama_kv_recency_store::get_migration_stats() const {
    if (data->layout != mode::committed_aging) {
        throw std::invalid_argument("migration statistics require committed recency storage");
    }
    migration_stats result;
    result.cold_pages = data->cold_pages;
    result.hot_pages = uint32_t(std::count_if(data->slot_page.begin(), data->slot_page.end(), [](int32_t page) { return page >= 0; }));
    const size_t cold_tokens = size_t(data->cold_pages)*data->plan.page_tokens;
    result.migrated_source_bytes = cold_tokens*data->plan.q8_bytes_per_token;
    result.resident_payload_bytes = cold_tokens*data->plan.q4_bytes_per_token +
        (data->next_cell - cold_tokens)*data->plan.q8_bytes_per_token;
    result.peak_workspace_bytes = data->peak_workspace_bytes;
    return result;
}

static const uint32_t RECENCY_STATE_MAGIC   = 0x564b5245; // "ERKV"
static const uint32_t RECENCY_STATE_VERSION = 1;

void llama_kv_recency_store::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    const auto & d = *data;
    if (d.layout != mode::committed_aging) {
        throw std::invalid_argument("KV recency state requires committed recency storage");
    }
    const bool held = d.next_cell && (seq_id < 0 || seq_id == d.sequence);
    const auto geometry = d.geometry();
    const uint32_t head[3] = {RECENCY_STATE_MAGIC, RECENCY_STATE_VERSION, uint32_t(geometry.size())};
    io.write(head, sizeof(head));
    io.write(geometry.data(), geometry.size()*sizeof(uint32_t));
    const int32_t cells[4] = {int32_t(held ? d.next_cell : 0), int32_t(held ? d.cold_pages : 0),
                              held ? d.first_position : -1, held ? d.sequence : -1};
    io.write(cells, sizeof(cells));
    if (held) {
        // ring slots are saved so a restore reproduces the attention row order exactly
        std::vector<uint32_t> slots;
        for (uint32_t page = d.cold_pages; page*d.plan.page_tokens < d.next_cell; ++page) {
            slots.push_back(uint32_t(d.page_slot[page]));
        }
        io.write(slots.data(), slots.size()*sizeof(uint32_t));
        d.for_each_row([&](ggml_tensor * t, size_t offset, size_t size) { io.write_tensor(t, offset, size); });
    }
}

llama_kv_recency_store::state_header llama_kv_recency_store::state_read_header(llama_io_read_i & io) const {
    const auto & d = *data;
    if (d.layout != mode::committed_aging) {
        throw std::invalid_argument("KV recency state requires committed recency storage");
    }
    uint32_t head[3] = {};
    io.read(head, sizeof(head));
    if (head[0] != RECENCY_STATE_MAGIC) {
        throw std::runtime_error("state holds no KV recency section");
    }
    if (head[1] != RECENCY_STATE_VERSION) {
        throw std::runtime_error("unsupported KV recency state version " + std::to_string(head[1]));
    }
    const auto geometry = d.geometry();
    if (head[2] != geometry.size()) {
        throw std::runtime_error("KV recency state was saved with a different layer layout");
    }
    std::vector<uint32_t> saved(geometry.size());
    io.read(saved.data(), saved.size()*sizeof(uint32_t));
    if (saved != geometry) {
        throw std::runtime_error("KV recency state was saved with a different session geometry");
    }
    int32_t cells[4] = {};
    io.read(cells, sizeof(cells));
    state_header result;
    result.n_cells = uint32_t(cells[0]);
    result.cold_pages = uint32_t(cells[1]);
    result.first_position = cells[2];
    result.sequence = cells[3];
    const uint32_t page_tokens = d.plan.page_tokens;
    const uint32_t pages = (result.n_cells + page_tokens - 1)/page_tokens;
    if (cells[0] < 0 || cells[1] < 0 || result.n_cells > d.plan.logical_ctx || result.cold_pages*page_tokens > result.n_cells ||
            result.cold_pages > d.plan.cold_tokens/page_tokens || pages - result.cold_pages > d.slot_page.size() ||
            (result.n_cells && (result.first_position < 0 || result.sequence < 0 || result.sequence >= LLAMA_MAX_SEQ))) {
        throw std::runtime_error("KV recency state has an invalid cell layout");
    }
    result.slots.resize(pages - result.cold_pages);
    io.read(result.slots.data(), result.slots.size()*sizeof(uint32_t));
    std::vector<bool> taken(d.slot_page.size());
    for (const uint32_t slot : result.slots) {
        if (slot >= taken.size() || taken[slot]) {
            throw std::runtime_error("KV recency state has an invalid ring layout");
        }
        taken[slot] = true;
    }
    return result;
}

void llama_kv_recency_store::state_read_rows(llama_io_read_i & io, const state_header & header, llama_seq_id seq_id) {
    auto & d = *data;
    truncate(0);
    if (!header.n_cells) {
        return;
    }
    const uint32_t page_tokens = d.plan.page_tokens;
    // q8_0 pages return to their saved ring slots; q4_0 pages keep their logical rows
    for (uint32_t page = header.cold_pages; page*page_tokens < header.n_cells; ++page) {
        const uint32_t slot = header.slots[page - header.cold_pages];
        d.allocate(1, slot*page_tokens/d.slab_rows[1]);
        d.page_slot[page] = int32_t(slot);
        d.slot_page[slot] = int32_t(page);
    }
    for (uint32_t row = 0; row < header.cold_pages*page_tokens; row += d.slab_rows[0]) {
        d.allocate(0, row/d.slab_rows[0]);
    }
    const llama_seq_id seq = seq_id >= 0 ? seq_id : header.sequence;
    for (uint32_t i = 0; i < header.n_cells; ++i) {
        d.cells.pos_set(i, header.first_position + llama_pos(i));
        d.cells.seq_add(i, seq);
    }
    d.next_cell = header.n_cells;
    d.cold_pages = header.cold_pages;
    d.first_position = header.first_position;
    d.sequence = seq;
    d.for_each_row([&](ggml_tensor * t, size_t offset, size_t size) { io.read_tensor(t, offset, size); });
}

//
// llama_kv_cache_recency_live
//

llama_kv_cache_recency_live::llama_kv_cache_recency_live(const llama_model & model, const std::vector<uint32_t> & layer_ids,
                                                         const llama_kv_recency_plan_params & params, std::vector<ggml_backend_t> readers) :
    hparams(model.hparams), store(params, llama_kv_recency_store::mode::committed_aging), readers(std::move(readers)) {
    if (layer_ids.size() != params.layers.size()) {
        throw std::invalid_argument("KV recency layer map does not match the plan");
    }
    int64_t head_k = -1;
    int64_t head_v = -1;
    for (size_t i = 0; i < layer_ids.size(); ++i) {
        const uint32_t il = layer_ids[i];
        layer_map[il] = uint32_t(i);
        head_k = head_k < 0 || head_k == hparams.n_embd_head_k(il) ? hparams.n_embd_head_k(il) : 0;
        head_v = head_v < 0 || head_v == hparams.n_embd_head_v(il) ? hparams.n_embd_head_v(il) : 0;
    }
    // same Hadamard rotation rule as the ordinary quantized cache, so Q8 rows match it
    const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
    if (!(LLAMA_ATTN_ROT_DISABLE && atoi(LLAMA_ATTN_ROT_DISABLE))) {
        if (head_k > 0 && hparams.n_embd_head_k() % 64 == 0) {
            n_rot_k = 64;
            while (head_k % (2*n_rot_k) == 0) {
                n_rot_k *= 2;
            }
        }
        if (head_v > 0 && hparams.n_embd_head_v() % 64 == 0) {
            n_rot_v = 64;
        }
    }
    for (const uint32_t n : {n_rot_k, n_rot_v}) {
        if (n && !hadamard.count(n)) {
            hadamard[n] = llama_kv_cache_hadamard(n);
        }
    }
}

llama_kv_cache_recency_live::~llama_kv_cache_recency_live() = default;

void llama_kv_cache_recency_live::synchronize() const {
    for (auto * backend : readers) {
        ggml_backend_synchronize(backend);
    }
}

bool llama_kv_cache_recency_live::seq_range(llama_seq_id seq_id, llama_pos & first, llama_pos & last) const {
    const uint32_t n = store.get_n_cells();
    if (!n || (seq_id >= 0 && !store.get_cells().seq_has(0, seq_id))) {
        return false;
    }
    first = store.get_cells().pos_get(0);
    last = store.get_cells().pos_get(n - 1);
    return true;
}

llama_memory_context_ptr llama_kv_cache_recency_live::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);
    const uint32_t page = store.get_plan().page_tokens;
    uint32_t next = store.get_n_cells();
    llama_pos last = -1;
    llama_seq_id seq = -1;
    if (next) {
        last = store.get_cells().pos_get(next - 1);
        for (seq = 0; seq < LLAMA_MAX_SEQ && !store.get_cells().seq_has(0, seq); ++seq) {}
    }
    balloc.split_reset();
    std::vector<llama_ubatch> ubatches;
    while (true) {
        // one ubatch never crosses a page, so it writes one Q8 slab and needs at most one new ring slot
        auto ubatch = balloc.split_simple(std::min(n_ubatch, page - next % page));
        if (!ubatch.n_tokens) {
            break;
        }
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            const llama_seq_id s = ubatch.seq_id[i][0];
            if (ubatch.n_pos != 1 || ubatch.n_seq_id[i] != 1 || (seq >= 0 && s != seq) ||
                    (last >= 0 && ubatch.pos[i] != last + 1)) {
                LLAMA_LOG_ERROR("%s: KV recency supports one sequence with consecutive positions\n", __func__);
                return std::make_unique<llama_kv_cache_recency_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }
            seq = s;
            last = ubatch.pos[i];
        }
        next += ubatch.n_tokens;
        ubatches.push_back(std::move(ubatch));
    }
    if (balloc.get_n_used() < balloc.get_n_tokens() || next > store.get_plan().logical_ctx) {
        return std::make_unique<llama_kv_cache_recency_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    return std::make_unique<llama_kv_cache_recency_context>(this, std::move(ubatches));
}

llama_memory_context_ptr llama_kv_cache_recency_live::init_full() {
    return std::make_unique<llama_kv_cache_recency_context>(this);
}

llama_memory_context_ptr llama_kv_cache_recency_live::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_kv_cache_recency_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_kv_cache_recency_live::get_can_shift() const {
    return false;
}

void llama_kv_cache_recency_live::clear(bool data) {
    GGML_UNUSED(data);
    synchronize();
    store.truncate(0);
}

bool llama_kv_cache_recency_live::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    llama_pos first = 0;
    llama_pos last = 0;
    if (!seq_range(seq_id, first, last)) {
        return true;
    }
    p0 = p0 < 0 ? first : p0;
    p1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    if (p0 >= p1 || p1 <= first || p0 > last) {
        return true;
    }
    if (p1 <= last) {
        return false; // only the tail can be removed
    }
    synchronize();
    return store.truncate(uint32_t(std::max(p0, first) - first));
}

void llama_kv_cache_recency_live::seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    llama_pos first = 0;
    llama_pos last = 0;
    if (src == dst || !seq_range(src, first, last) || (p1 >= 0 && p1 <= first) || p0 > last) {
        return;
    }
    GGML_ABORT("KV recency holds one sequence; seq_cp is not supported");
}

void llama_kv_cache_recency_live::seq_keep(llama_seq_id seq_id) {
    llama_pos first = 0;
    llama_pos last = 0;
    if (store.get_n_cells() && !seq_range(seq_id, first, last)) {
        clear(true);
    }
}

void llama_kv_cache_recency_live::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_pos first = 0;
    llama_pos last = 0;
    if (shift == 0 || !seq_range(seq_id, first, last) || (p1 >= 0 && p1 <= first) || p0 > last) {
        return;
    }
    GGML_ABORT("KV recency does not support position shifts");
}

void llama_kv_cache_recency_live::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_pos first = 0;
    llama_pos last = 0;
    if (d == 1 || !seq_range(seq_id, first, last) || (p1 >= 0 && p1 <= first) || p0 > last) {
        return;
    }
    GGML_ABORT("KV recency does not support position division");
}

llama_pos llama_kv_cache_recency_live::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos first = -1;
    llama_pos last = -1;
    return seq_range(seq_id, first, last) ? first : -1;
}

llama_pos llama_kv_cache_recency_live::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos first = -1;
    llama_pos last = -1;
    return seq_range(seq_id, first, last) ? last : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_recency_live::memory_breakdown() const {
    return store.memory_breakdown();
}

void llama_kv_cache_recency_live::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // partial state covers recurrent memory only
    if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        return;
    }
    synchronize();
    store.state_write(io, seq_id);
}

void llama_kv_cache_recency_live::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        return;
    }
    const auto header = store.state_read_header(io);
    synchronize();
    try {
        store.state_read_rows(io, header, seq_id);
    } catch (...) {
        store.truncate(0);
        throw;
    }
}

//
// llama_kv_cache_recency_context
//

llama_kv_cache_recency_context::llama_kv_cache_recency_context(llama_memory_status status) : status(status) {}

llama_kv_cache_recency_context::llama_kv_cache_recency_context(llama_kv_cache_recency_live * mem) :
    status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem) {
    // graph reservation needs one Q8 destination slab
    mem->get_store().allocate_slab(GGML_TYPE_Q8_0, 0);
}

llama_kv_cache_recency_context::llama_kv_cache_recency_context(llama_kv_cache_recency_live * mem, std::vector<llama_ubatch> ubatches) :
    status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_kv_cache_recency_context::~llama_kv_cache_recency_context() = default;

bool llama_kv_cache_recency_context::next() {
    return ++i_cur < ubatches.size();
}

bool llama_kv_cache_recency_context::apply() {
    if (!mem || ubatches.empty()) {
        return true;
    }
    const auto & ubatch = ubatches[i_cur];
    auto & store = mem->get_store();
    try {
        try {
            first_cell = store.append(ubatch.pos[0], ubatch.seq_id[0][0], ubatch.n_tokens);
        } catch (const std::runtime_error &) {
            // full ring: the oldest eligible pages age to Q4, then the page fits
            if (!store.migrate_oldest_pages(mem->get_readers(), UINT32_MAX)) {
                throw;
            }
            first_cell = store.append(ubatch.pos[0], ubatch.seq_id[0][0], ubatch.n_tokens);
        }
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: KV recency could not commit the ubatch: %s\n", __func__, error.what());
        return false;
    }
    dst_slab = store.get_q8_row(first_cell).first;
    return true;
}

llama_memory_status llama_kv_cache_recency_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_recency_context::get_ubatch() const {
    GGML_ASSERT(status == LLAMA_MEMORY_STATUS_SUCCESS && i_cur < ubatches.size());
    return ubatches[i_cur];
}

std::vector<llama_kv_cache_recency_context::region> llama_kv_cache_recency_context::get_regions() const {
    std::vector<region> result;
    const auto & store = mem->get_store();
    for (ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        for (uint32_t slab = 0; slab < store.get_slab_count(type); ++slab) {
            if (store.get_tensor(0, true, type, slab)) {
                result.push_back({type, slab});
            }
        }
    }
    return result;
}

uint32_t llama_kv_cache_recency_context::get_rows(const region & r) const {
    return uint32_t(mem->get_store().get_tensor(0, true, r.type, r.slab)->ne[1]);
}

static ggml_tensor * recency_view(ggml_context * ctx, ggml_tensor * t, int64_t n_embd_head, int64_t n_head) {
    return ggml_view_4d(ctx, t, n_embd_head, n_head, t->ne[1], 1, ggml_row_size(t->type, n_embd_head), t->nb[1], t->nb[2], 0);
}

ggml_tensor * llama_kv_cache_recency_context::get_k(ggml_context * ctx, int32_t il, const region & r) const {
    const auto & hparams = mem->get_hparams();
    return recency_view(ctx, mem->get_store().get_tensor(mem->get_layer(il), true, r.type, r.slab), hparams.n_embd_head_k(il), hparams.n_head_kv(il));
}

ggml_tensor * llama_kv_cache_recency_context::get_v(ggml_context * ctx, int32_t il, const region & r) const {
    const auto & hparams = mem->get_hparams();
    return recency_view(ctx, mem->get_store().get_tensor(mem->get_layer(il), false, r.type, r.slab), hparams.n_embd_head_v(il), hparams.n_head_kv(il));
}

static ggml_tensor * recency_cpy(ggml_context * ctx, ggml_tensor * dst, ggml_tensor * cur, ggml_tensor * idxs) {
    // merge dims 0 and 1 as the ordinary cache does
    GGML_ASSERT(ggml_row_size(cur->type, cur->ne[0]) == cur->nb[1]);
    cur = ggml_view_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2], cur->nb[2], 0);
    return ggml_set_rows(ctx, dst, cur, idxs);
}

ggml_tensor * llama_kv_cache_recency_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il) const {
    return recency_cpy(ctx, mem->get_store().get_tensor(mem->get_layer(il), true, GGML_TYPE_Q8_0, dst_slab), k_cur, idxs);
}

ggml_tensor * llama_kv_cache_recency_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il) const {
    return recency_cpy(ctx, mem->get_store().get_tensor(mem->get_layer(il), false, GGML_TYPE_Q8_0, dst_slab), v_cur, idxs);
}

uint32_t llama_kv_cache_recency_context::get_n_rot_k() const {
    return mem->get_n_rot_k();
}

uint32_t llama_kv_cache_recency_context::get_n_rot_v() const {
    return mem->get_n_rot_v();
}

void llama_kv_cache_recency_context::set_input_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    llama_host_write(dst);
    auto * data = (int64_t *) dst->data;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const auto row = mem->get_store().get_q8_row(first_cell + i);
        GGML_ASSERT(row.first == dst_slab);
        data[i] = row.second;
    }
}

void llama_kv_cache_recency_context::set_input_kq_mask(ggml_tensor * dst, const region & r, const llama_ubatch * ubatch) const {
    llama_host_write(dst);
    const auto & store = mem->get_store();
    const auto & cells = store.get_cells();
    const bool alibi = mem->get_hparams().use_alibi;
    const int64_t n_rows = dst->ne[0];
    std::vector<llama_pos> pos(n_rows, -1);
    std::vector<llama_seq_id> seq(n_rows, -1);
    for (int64_t row = 0; row < n_rows; ++row) {
        const int32_t cell = store.get_row_cell(r.type, r.slab, uint32_t(row));
        if (cell >= 0) {
            pos[row] = cells.pos_get(cell);
            for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ && seq[row] < 0; ++s) {
                seq[row] = cells.seq_has(cell, s) ? s : -1;
            }
        }
    }
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    auto * data = (ggml_fp16_t *) dst->data;
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos p1 = ubatch->pos[i];
        const llama_seq_id s1 = ubatch->seq_id[i][0];
        for (int64_t row = 0; row < n_rows; ++row) {
            const bool keep = pos[row] >= 0 && seq[row] == s1 && pos[row] <= p1;
            data[i*n_rows + row] = keep ? ggml_fp32_to_fp16(alibi ? -float(std::abs(pos[row] - p1)) : 0.0f) : drop;
        }
    }
}

void llama_kv_cache_recency_context::set_input_rot(ggml_tensor * dst) const {
    llama_host_write(dst);
    const auto & rot = mem->get_rot(uint32_t(dst->ne[0]));
    memcpy(dst->data, rot.data(), ggml_nbytes(dst));
}
