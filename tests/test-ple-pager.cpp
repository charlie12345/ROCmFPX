#include "llama-ple-pager.h"

#include "ggml.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#endif

int main() {
    const llama_ple_source source {
        /* tensor_offset = */ 547040384,
        /* tensor_bytes  = */ 54400261120,
        /* row_count     = */ 320001536,
        /* row_elements  = */ 160,
        /* row_bytes     = */ 170,
        /* alignment     = */ 4096,
    };

    const auto plan = llama_ple_plan_rows(source, {0, 5, 5, 6}, 32*1024*1024);
    assert(plan.rows == std::vector<int32_t>({0, 5, 5, 6}));
    assert(plan.row_spans.size() == plan.rows.size());
    assert(plan.sectors.size() < 5);
    assert(plan.peak_bytes <= 32*1024*1024);
    assert(plan.waves.size() == 1);

    const auto one_sector = llama_ple_plan_rows(source, {0, 1, 2}, 4096);
    assert(one_sector.waves.size() == one_sector.sectors.size());
    for (const auto & wave : one_sector.waves) {
        assert(wave.bytes == 4096);
    }

    const llama_ple_source boundary {
        /* tensor_offset = */ 4095,
        /* tensor_bytes  = */ 8,
        /* row_count     = */ 2,
        /* row_elements  = */ 1,
        /* row_bytes     = */ 4,
        /* alignment     = */ 4096,
    };
    const auto boundary_plan = llama_ple_plan_rows(boundary, {0}, 4096*2);
    assert(boundary_plan.sectors.size() == 2);
    assert(boundary_plan.row_spans[0].sector_count == 2);

    auto bad_row_count = [&]() {
        (void) llama_ple_plan_rows(boundary, {2}, 4096);
    };
    bool threw = false;
    try {
        bad_row_count();
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw);

    auto overflow = [&]() {
        const llama_ple_source bad {
            std::numeric_limits<uint64_t>::max() - 1,
            4, 1, 1, 4, 4096,
        };
        (void) llama_ple_plan_rows(bad, {0}, 4096);
    };
    threw = false;
    try {
        overflow();
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw);

    std::vector<float> input(160);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin((float) i * 0.17f) * 12.0f;
    }
    std::vector<uint8_t> quantized(170);
    assert(ggml_quantize_chunk(GGML_TYPE_Q8_0, input.data(), quantized.data(), 0, 1, 160, nullptr) == quantized.size());

    std::vector<float> expected(160);
    std::vector<float> actual(160);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    assert(traits != nullptr && traits->to_float != nullptr);
    traits->to_float(quantized.data(), expected.data(), expected.size());
    llama_ple_dequantize_q8_0_rows(quantized.data(), 1, 170, 160, actual.data(), actual.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(std::abs(actual[i] - expected[i]) < 1e-6f);
    }

#ifdef _WIN32
    {
        const std::wstring path = L"C:\\llama-ple-pager-test-" + std::to_wstring(GetCurrentProcessId()) + L".bin";
        HANDLE writer = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
        assert(writer != INVALID_HANDLE_VALUE);

        constexpr uint64_t tensor_offset = 4096;
        constexpr size_t row_elements = 160;
        constexpr size_t row_bytes = 170;
        constexpr size_t row_count = 40;
        std::vector<uint8_t> file_bytes(16384, 0);
        std::vector<float> row(row_elements);
        std::vector<uint8_t> quantized(row_bytes);
        std::vector<float> expected_rows(row_count * row_elements);
        for (size_t r = 0; r < row_count; ++r) {
            for (size_t i = 0; i < row_elements; ++i) {
                row[i] = (float) (r * 0.25 + i * 0.01);
            }
            assert(ggml_quantize_chunk(GGML_TYPE_Q8_0, row.data(), quantized.data(), 0, 1, row_elements, nullptr) == row_bytes);
            std::copy(quantized.begin(), quantized.end(), file_bytes.begin() + tensor_offset + r * row_bytes);
            const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
            traits->to_float(quantized.data(), expected_rows.data() + r * row_elements, row_elements);
        }
        DWORD written = 0;
        assert(WriteFile(writer, file_bytes.data(), (DWORD) file_bytes.size(), &written, nullptr));
        assert(written == file_bytes.size());
        assert(CloseHandle(writer));

        const int fd = _wopen(path.c_str(), _O_RDONLY | _O_BINARY);
        assert(fd >= 0);
        const llama_ple_source test_source {
            tensor_offset,
            row_count * row_bytes,
            row_count,
            row_elements,
            row_bytes,
            4096,
        };
        auto pager = llama_ple_pager::open_from_file_id(fd, test_source, 4, 8192);
        _close(fd);

        const std::vector<int32_t> requested = {0, 24, 24, 25};
        std::vector<float> output(requested.size() * row_elements);
        pager->read_rows(requested, output.data(), output.size());
        for (size_t n = 0; n < requested.size(); ++n) {
            const auto * expected = expected_rows.data() + (size_t) requested[n] * row_elements;
            for (size_t i = 0; i < row_elements; ++i) {
                assert(std::abs(output[n * row_elements + i] - expected[i]) < 1e-6f);
            }
        }
        bool invalid_threw = false;
        try {
            pager->read_rows({row_count}, output.data(), output.size());
        } catch (const std::exception &) {
            invalid_threw = true;
        }
        assert(invalid_threw);
        pager->read_rows({1}, output.data(), row_elements);
        assert(std::abs(output[0] - expected_rows[row_elements]) < 1e-6f);
        const auto stats = pager->snapshot_stats();
        assert(stats.rows == requested.size() + 1);
        assert(stats.bytes > 0);
        assert(stats.reads > 0);
        assert(stats.failures == 0);

        DeleteFileW(path.c_str());
    }
#endif

    return 0;
}
