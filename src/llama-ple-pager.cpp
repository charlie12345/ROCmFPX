#include "llama-ple-pager.h"

#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <mutex>
#include <system_error>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#endif

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

static uint64_t align_down(uint64_t value, uint64_t alignment) {
    return value - value % alignment;
}

static uint64_t align_up_checked(uint64_t value, uint64_t alignment) {
    const uint64_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    uint64_t result = 0;
    if (!checked_add_u64(value, alignment - remainder, result)) {
        throw std::overflow_error("PLE sector alignment overflow");
    }
    return result;
}

llama_ple_read_plan llama_ple_plan_rows(
        const llama_ple_source & source,
        const std::vector<int32_t> & rows,
        size_t buffer_bytes) {
    if (source.alignment == 0 || source.row_bytes == 0 || source.row_elements == 0) {
        throw std::invalid_argument("invalid PLE row geometry");
    }
    if (buffer_bytes == 0) {
        throw std::invalid_argument("PLE buffer must not be empty");
    }

    uint64_t tensor_end = 0;
    if (!checked_add_u64(source.tensor_offset, source.tensor_bytes, tensor_end)) {
        throw std::overflow_error("PLE tensor extent overflow");
    }

    llama_ple_read_plan result;
    result.rows = rows;
    result.row_spans.reserve(rows.size());
    std::map<uint64_t, uint64_t> sector_sizes;

    for (const int32_t row : rows) {
        if (row < 0 || (uint64_t) row >= source.row_count) {
            throw std::out_of_range("PLE row index out of range");
        }

        uint64_t row_delta = 0;
        uint64_t row_offset = 0;
        uint64_t row_end = 0;
        if (!checked_mul_u64((uint64_t) row, source.row_bytes, row_delta) ||
            !checked_add_u64(source.tensor_offset, row_delta, row_offset) ||
            !checked_add_u64(row_offset, source.row_bytes, row_end) ||
            row_end > tensor_end) {
            throw std::overflow_error("PLE row extent overflow");
        }

        const uint64_t sector_begin = align_down(row_offset, source.alignment);
        const uint64_t sector_end = align_up_checked(row_end, source.alignment);
        for (uint64_t sector = sector_begin; sector < sector_end;) {
            sector_sizes.emplace(sector, source.alignment);
            uint64_t next = 0;
            if (!checked_add_u64(sector, source.alignment, next) || next <= sector) {
                throw std::overflow_error("PLE sector range overflow");
            }
            sector = next;
        }

        result.row_spans.push_back({
            row_offset,
            row_offset - sector_begin,
            0,
            (size_t) ((sector_end - sector_begin) / source.alignment),
        });
    }

    result.sectors.reserve(sector_sizes.size());
    for (const auto & [offset, bytes] : sector_sizes) {
        result.sectors.push_back({offset, bytes});
    }

    for (auto & span : result.row_spans) {
        const auto it = std::lower_bound(
            result.sectors.begin(), result.sectors.end(), span.row_offset - span.first_byte,
            [](const llama_ple_sector & sector, uint64_t offset) { return sector.offset < offset; });
        if (it == result.sectors.end() || it->offset != span.row_offset - span.first_byte) {
            throw std::logic_error("PLE row sector was not planned");
        }
        span.sector_begin = (size_t) std::distance(result.sectors.begin(), it);
    }

    size_t wave_begin = 0;
    uint64_t wave_bytes = 0;
    for (size_t i = 0; i < result.sectors.size(); ++i) {
        const uint64_t sector_bytes = result.sectors[i].bytes;
        if (sector_bytes > buffer_bytes) {
            throw std::invalid_argument("PLE buffer is smaller than one aligned sector");
        }
        if (wave_bytes != 0 && wave_bytes > buffer_bytes - sector_bytes) {
            result.waves.push_back({wave_begin, i - wave_begin, wave_bytes});
            result.peak_bytes = std::max(result.peak_bytes, wave_bytes);
            wave_begin = i;
            wave_bytes = 0;
        }
        wave_bytes += sector_bytes;
    }
    if (wave_begin < result.sectors.size()) {
        result.waves.push_back({wave_begin, result.sectors.size() - wave_begin, wave_bytes});
        result.peak_bytes = std::max(result.peak_bytes, wave_bytes);
    }

    return result;
}

void llama_ple_dequantize_q8_0_rows(
        const uint8_t * input,
        size_t row_count,
        size_t row_bytes,
        size_t row_elements,
        float * output,
        size_t output_values) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("PLE dequantization requires non-null buffers");
    }
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    if (traits == nullptr || traits->to_float == nullptr) {
        throw std::runtime_error("Q8_0 dequantization trait is unavailable");
    }
    const size_t expected_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, row_elements);
    if (row_bytes != expected_row_bytes || output_values != row_count * row_elements) {
        throw std::invalid_argument(
            "invalid Q8_0 PLE row geometry: row_bytes=" + std::to_string(row_bytes) +
            ", expected=" + std::to_string(expected_row_bytes) +
            ", row_elements=" + std::to_string(row_elements) +
            ", row_count=" + std::to_string(row_count));
    }
    for (size_t row = 0; row < row_count; ++row) {
        traits->to_float(input + row * row_bytes, output + row * row_elements, row_elements);
    }
}

struct llama_ple_pager::impl {
    llama_ple_source source{};
    uint32_t io_depth = 0;
    size_t buffer_bytes = 0;
    llama_ple_pager_stats stats{};
    mutable std::mutex mutex;

#ifdef _WIN32
    struct lane {
        OVERLAPPED overlapped{};
        HANDLE event = nullptr;
        size_t sector_index = 0;
        size_t buffer_offset = 0;
        bool active = false;
    };

    HANDLE file = INVALID_HANDLE_VALUE;
    uint64_t file_bytes = 0;
    uint8_t * arena = nullptr;
    std::vector<lane> lanes;

    static std::runtime_error win_error(const char * operation, DWORD error = GetLastError()) {
        return std::runtime_error(std::string(operation) + ": " + std::system_category().message((int) error));
    }

    void cancel_and_drain() noexcept {
        if (file == INVALID_HANDLE_VALUE) {
            return;
        }
        for (auto & lane : lanes) {
            if (!lane.active) {
                continue;
            }
            CancelIoEx(file, &lane.overlapped);
            DWORD bytes = 0;
            GetOverlappedResult(file, &lane.overlapped, &bytes, TRUE);
            lane.active = false;
        }
    }

    ~impl() {
        cancel_and_drain();
        for (auto & lane : lanes) {
            if (lane.event != nullptr) {
                CloseHandle(lane.event);
            }
        }
        if (arena != nullptr) {
            VirtualFree(arena, 0, MEM_RELEASE);
        }
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
        }
    }

    void probe() {
        const uint64_t probe_offset = align_down(source.tensor_offset, source.alignment);
        uint64_t probe_end = 0;
        if (!checked_add_u64(probe_offset, source.alignment, probe_end) || probe_end > file_bytes) {
            throw std::runtime_error("PLE startup probe is outside the source file");
        }

        auto & lane = lanes.front();
        std::memset(&lane.overlapped, 0, sizeof(lane.overlapped));
        lane.overlapped.Offset = (DWORD) (probe_offset & 0xffffffffu);
        lane.overlapped.OffsetHigh = (DWORD) (probe_offset >> 32);
        lane.overlapped.hEvent = lane.event;
        ResetEvent(lane.event);
        DWORD bytes = 0;
        const BOOL ok = ReadFile(file, arena, (DWORD) source.alignment, &bytes, &lane.overlapped);
        if (!ok && GetLastError() != ERROR_IO_PENDING) {
            throw win_error("PLE startup probe");
        }
        if (!GetOverlappedResult(file, &lane.overlapped, &bytes, TRUE) || bytes != source.alignment) {
            throw win_error("PLE startup probe completion");
        }
    }

    void read_plan(const llama_ple_read_plan & plan, uint8_t * compact) {
        std::vector<size_t> arena_offsets(plan.sectors.size(), std::numeric_limits<size_t>::max());
        uint64_t total_read_bytes = 0;
        uint64_t total_reads = 0;
        const auto read_start = std::chrono::steady_clock::now();

        try {
            for (const auto & wave : plan.waves) {
                size_t offset = 0;
                for (size_t i = wave.sector_begin; i < wave.sector_begin + wave.sector_count; ++i) {
                    arena_offsets[i] = offset;
                    offset += (size_t) plan.sectors[i].bytes;
                }

                for (size_t next = wave.sector_begin; next < wave.sector_begin + wave.sector_count;) {
                    const size_t count = std::min(lanes.size(), wave.sector_begin + wave.sector_count - next);
                    for (size_t n = 0; n < count; ++n) {
                        auto & lane = lanes[n];
                        const size_t sector_index = next + n;
                        const auto & sector = plan.sectors[sector_index];
                        std::memset(&lane.overlapped, 0, sizeof(lane.overlapped));
                        lane.overlapped.Offset = (DWORD) (sector.offset & 0xffffffffu);
                        lane.overlapped.OffsetHigh = (DWORD) (sector.offset >> 32);
                        lane.overlapped.hEvent = lane.event;
                        lane.sector_index = sector_index;
                        lane.buffer_offset = arena_offsets[sector_index];
                        ResetEvent(lane.event);
                        DWORD bytes = 0;
                        const BOOL ok = ReadFile(file, arena + lane.buffer_offset,
                            (DWORD) sector.bytes, &bytes, &lane.overlapped);
                        if (!ok && GetLastError() != ERROR_IO_PENDING) {
                            throw win_error("PLE direct read");
                        }
                        lane.active = true;
                        ++total_reads;
                    }

                    for (size_t n = 0; n < count; ++n) {
                        auto & lane = lanes[n];
                        DWORD bytes = 0;
                        if (!GetOverlappedResult(file, &lane.overlapped, &bytes, TRUE) ||
                            bytes != plan.sectors[lane.sector_index].bytes) {
                            lane.active = false;
                            throw win_error("PLE direct read completion");
                        }
                        lane.active = false;
                        total_read_bytes += bytes;
                    }

                    for (size_t row_index = 0; row_index < plan.row_spans.size(); ++row_index) {
                        const auto & span = plan.row_spans[row_index];
                        uint64_t row_end = 0;
                        if (!checked_add_u64(span.row_offset, source.row_bytes, row_end)) {
                            throw std::overflow_error("PLE row scatter overflow");
                        }
                        for (size_t i = wave.sector_begin; i < wave.sector_begin + wave.sector_count; ++i) {
                            const auto & sector = plan.sectors[i];
                            uint64_t sector_end = 0;
                            if (!checked_add_u64(sector.offset, sector.bytes, sector_end)) {
                                throw std::overflow_error("PLE sector scatter overflow");
                            }
                            const uint64_t begin = std::max(span.row_offset, sector.offset);
                            const uint64_t end = std::min(row_end, sector_end);
                            if (begin < end) {
                                const size_t destination = row_index * (size_t) source.row_bytes + (size_t) (begin - span.row_offset);
                                const size_t input = arena_offsets[i] + (size_t) (begin - sector.offset);
                                std::memcpy(compact + destination, arena + input, (size_t) (end - begin));
                            }
                        }
                    }
                    next += count;
                }
            }
        } catch (...) {
            cancel_and_drain();
            std::lock_guard<std::mutex> lock(mutex);
            ++stats.failures;
            throw;
        }

        const auto read_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - read_start).count();
        std::lock_guard<std::mutex> lock(mutex);
        stats.bytes += total_read_bytes;
        stats.reads += total_reads;
        stats.read_us += read_us;
        stats.max_read_us = std::max(stats.max_read_us, read_us);
    }
#else
    ~impl() = default;
#endif
};

llama_ple_pager::llama_ple_pager(std::unique_ptr<impl> pimpl) : pimpl_(std::move(pimpl)) {}

llama_ple_pager::~llama_ple_pager() = default;

std::unique_ptr<llama_ple_pager> llama_ple_pager::open_from_file_id(
        int file_id,
        const llama_ple_source & source,
        uint32_t io_depth,
        size_t buffer_bytes) {
#ifdef _WIN32
    if (file_id < 0 || io_depth == 0 || buffer_bytes == 0 || source.alignment == 0) {
        throw std::invalid_argument("invalid PLE pager open parameters");
    }
    auto pager_impl = std::make_unique<impl>();
    pager_impl->source = source;
    pager_impl->io_depth = io_depth;
    pager_impl->buffer_bytes = buffer_bytes;

    const intptr_t raw_handle = _get_osfhandle(file_id);
    if (raw_handle == -1) {
        throw std::runtime_error("PLE source file descriptor has no Windows handle");
    }
    const HANDLE source_handle = (HANDLE) raw_handle;
    DWORD path_length = GetFinalPathNameByHandleW(source_handle, nullptr, 0, FILE_NAME_NORMALIZED);
    if (path_length == 0) {
        throw impl::win_error("PLE source path lookup");
    }
    std::wstring path(path_length, L'\0');
    path_length = GetFinalPathNameByHandleW(source_handle, path.data(), (DWORD) path.size(), FILE_NAME_NORMALIZED);
    if (path_length == 0 || path_length >= path.size()) {
        throw impl::win_error("PLE source path lookup");
    }
    path.resize(path_length);

    pager_impl->file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                   FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (pager_impl->file == INVALID_HANDLE_VALUE) {
        throw impl::win_error("PLE direct handle open");
    }

    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(pager_impl->file, &file_size) || file_size.QuadPart < 0) {
        throw impl::win_error("PLE source size query");
    }
    pager_impl->file_bytes = (uint64_t) file_size.QuadPart;

    FILE_ALIGNMENT_INFO alignment_info{};
    if (!GetFileInformationByHandleEx(pager_impl->file, FileAlignmentInfo,
                                      &alignment_info, sizeof(alignment_info))) {
        throw impl::win_error("PLE alignment query");
    }
    const uint64_t required_alignment = (uint64_t) alignment_info.AlignmentRequirement + 1;
    if (required_alignment == 0 || source.alignment % required_alignment != 0) {
        throw std::runtime_error("PLE source alignment is incompatible with the direct handle");
    }

    pager_impl->arena = (uint8_t *) VirtualAlloc(nullptr, buffer_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (pager_impl->arena == nullptr) {
        throw impl::win_error("PLE staging allocation");
    }
    const size_t max_lanes = std::max<size_t>(1, buffer_bytes / (size_t) source.alignment);
    const size_t lane_count = std::min<size_t>(io_depth, max_lanes);
    pager_impl->lanes.resize(lane_count);
    for (auto & lane : pager_impl->lanes) {
        lane.event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (lane.event == nullptr) {
            throw impl::win_error("PLE event creation");
        }
    }
    pager_impl->probe();
    return std::unique_ptr<llama_ple_pager>(new llama_ple_pager(std::move(pager_impl)));
#else
    GGML_UNUSED(file_id);
    GGML_UNUSED(source);
    GGML_UNUSED(io_depth);
    GGML_UNUSED(buffer_bytes);
    throw std::runtime_error("PLE direct pager is only available on Windows");
#endif
}

void llama_ple_pager::read_rows(const std::vector<int32_t> & rows, float * output, size_t output_values) {
    if (rows.empty()) {
        if (output_values != 0) {
            throw std::invalid_argument("empty PLE row request has non-empty output");
        }
        return;
    }
    if (output == nullptr) {
        throw std::invalid_argument("PLE output buffer is null");
    }
    if (pimpl_->source.row_elements > std::numeric_limits<size_t>::max() ||
        rows.size() > std::numeric_limits<size_t>::max() / (size_t) pimpl_->source.row_elements ||
        output_values != rows.size() * (size_t) pimpl_->source.row_elements) {
        throw std::invalid_argument("invalid PLE output geometry");
    }

    const auto plan = llama_ple_plan_rows(pimpl_->source, rows, pimpl_->buffer_bytes);
#ifdef _WIN32
    std::vector<uint8_t> compact(rows.size() * (size_t) pimpl_->source.row_bytes);
    pimpl_->read_plan(plan, compact.data());
    const auto dequant_start = std::chrono::steady_clock::now();
    llama_ple_dequantize_q8_0_rows(compact.data(), rows.size(),
        (size_t) pimpl_->source.row_bytes, (size_t) pimpl_->source.row_elements,
        output, output_values);
    const auto dequant_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - dequant_start).count();
    {
        std::lock_guard<std::mutex> lock(pimpl_->mutex);
        pimpl_->stats.rows += rows.size();
        pimpl_->stats.sectors += plan.sectors.size();
        uint64_t references = 0;
        for (const auto & span : plan.row_spans) {
            references += span.sector_count;
        }
        pimpl_->stats.deduplicated_sectors += references >= plan.sectors.size() ? references - plan.sectors.size() : 0;
        pimpl_->stats.dequant_us += dequant_us;
        pimpl_->stats.peak_arena_bytes = std::max(pimpl_->stats.peak_arena_bytes, plan.peak_bytes);
    }
#else
    GGML_UNUSED(plan);
    GGML_UNUSED(output);
    GGML_UNUSED(output_values);
    throw std::runtime_error("PLE direct pager is only available on Windows");
#endif
}

llama_ple_pager_stats llama_ple_pager::snapshot_stats() const {
    std::lock_guard<std::mutex> lock(pimpl_->mutex);
    return pimpl_->stats;
}
