#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#define HIP_CHECK(expr) do { const hipError_t err_ = (expr); if (err_ != hipSuccess) { std::fprintf(stderr, "HIP failure %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err_)); std::exit(2); } } while (0)

static constexpr int GATE_K = 4096;
static constexpr int GATE_ROWS = 2048;
static constexpr int DOWN_K = 2048;
static constexpr int DOWN_ROWS = 4096;
static constexpr int MMQ_NS[] = {9, 512, 2048};
static constexpr int THREADS = 128;
static constexpr uint32_t CANDIDATE_UE4M4FAST_PRIVATE = 0x55344d46u;
static constexpr int8_t UE4M4FAST_CODEBOOK[16] = {0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10};

struct UE4M4FastWire { uint8_t bytes[17]; };
struct Q8Block { float d; int8_t qs[32]; };
static_assert(sizeof(UE4M4FastWire) == 17);
static_assert(sizeof(Q8Block) == 36);

struct DeviceCounters {
    uint32_t dispatch_count;
    uint32_t invalid_count;
    uint32_t fallback_count;
    uint32_t inactive_slots;
};

struct RouteSlot { int32_t expert_id; int32_t active; };

struct ue4m4fast_device_oracle_case {
    const char * case_id;
    uint32_t source_bits[32];
    uint8_t wire[17];
    uint32_t reconstruction_bits[32];
};
#include "ue4m4fast-source-matched-device-oracle-v1.inc"

static constexpr int UE4M4_LEGAL_METADATA_BYTES = 256;
static constexpr int UE4M4_INVALID_METADATA_BYTES = 0;
static constexpr const char * UE4M4_MALFORMED_WIRE_CONTRACT = "NO_INVALID_METADATA_ALL_256_UE4M4_BYTES_LEGAL";

static __host__ __device__ bool decode_scale(uint8_t metadata, float & value) {
    const int exponent = metadata >> 4;
    const int mantissa = metadata & 15;
    value = exponent == 0 ? ldexpf(static_cast<float>(mantissa), -11)
                          : ldexpf(static_cast<float>(16 + mantissa), exponent - 12);
    return true;
}

static __host__ __device__ int8_t codebook_value(uint8_t code) {
    return UE4M4FAST_CODEBOOK[code & 15u];
}

static __device__ __forceinline__ bool direct_dot_block(
        const UE4M4FastWire & wire, const Q8Block & activation, float & value) {
    float scale = 0.0f;
    if (!decode_scale(wire.bytes[16], scale)) return false;
    int sum = 0;
    for (int weight = 0; weight < 32; ++weight) {
        const uint8_t packed = wire.bytes[weight >> 1];
        const uint8_t code = ((weight & 1) ? (packed >> 4) : packed) & 15u;
        sum += static_cast<int>(codebook_value(code)) * static_cast<int>(activation.qs[weight]);
    }
    value = activation.d * scale * static_cast<float>(sum);
    return true;
}

static __device__ __forceinline__ void count_workgroup_dispatch(DeviceCounters * counters, int rows) {
    if (threadIdx.x == 0) {
        const int first = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
        const int remaining = rows - first;
        if (remaining > 0) atomicAdd(&counters->dispatch_count, static_cast<uint32_t>(remaining < static_cast<int>(blockDim.x) ? remaining : static_cast<int>(blockDim.x)));
    }
}

__global__ void ue4m4fast_deepseek_mmvq_kernel(
        const UE4M4FastWire * wires, const Q8Block * activations, float * output,
        uint32_t * route_ids, DeviceCounters * counters, int rows, int k_blocks) {
    count_workgroup_dispatch(counters, rows);
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= rows) return;
    float total = 0.0f;
    bool valid = true;
    for (int block = 0; block < k_blocks; ++block) {
        float partial = 0.0f;
        if (!direct_dot_block(wires[static_cast<size_t>(row) * k_blocks + block], activations[block], partial)) {
            valid = false;
            break;
        }
        total += partial;
    }
    route_ids[row] = CANDIDATE_UE4M4FAST_PRIVATE;
    if (!valid) {
        atomicAdd(&counters->invalid_count, 1u);
        output[row] = 0.0f;
        return;
    }
    output[row] = total;
}

__global__ void ue4m4fast_deepseek_mmq_kernel(
        const UE4M4FastWire * wires, const Q8Block * activations, float * output,
        uint32_t * route_ids, DeviceCounters * counters, int rows, int n, int k_blocks) {
    count_workgroup_dispatch(counters, rows);
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    if (row >= rows || col >= n) return;
    float total = 0.0f;
    bool valid = true;
    for (int block = 0; block < k_blocks; ++block) {
        float partial = 0.0f;
        if (!direct_dot_block(
                wires[static_cast<size_t>(row) * k_blocks + block],
                activations[static_cast<size_t>(col) * k_blocks + block], partial)) {
            valid = false;
            break;
        }
        total += partial;
    }
    const size_t index = static_cast<size_t>(col) * rows + row;
    route_ids[index] = CANDIDATE_UE4M4FAST_PRIVATE;
    if (!valid) {
        atomicAdd(&counters->invalid_count, 1u);
        output[index] = 0.0f;
        return;
    }
    output[index] = total;
}

__global__ void ue4m4fast_routed_semantics_kernel(
        const UE4M4FastWire * expert_wires, const Q8Block * activation, const RouteSlot * slots,
        float * output, uint32_t * route_ids, int32_t * expert_ids, DeviceCounters * counters,
        int rows, int slot_count, int expert_count, int k_blocks) {
    const int slot = static_cast<int>(blockIdx.y);
    if (slot >= slot_count) return;
    const RouteSlot route = slots[slot];
    if (!route.active) {
        if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&counters->inactive_slots, 1u);
        return;
    }
    count_workgroup_dispatch(counters, rows);
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= rows) return;
    const size_t index = static_cast<size_t>(slot) * rows + row;
    if (route.expert_id < 0 || route.expert_id >= expert_count) {
        atomicAdd(&counters->invalid_count, 1u);
        output[index] = 0.0f;
        return;
    }
    float total = 0.0f;
    bool valid = true;
    for (int block = 0; block < k_blocks; ++block) {
        const size_t wire_index = (static_cast<size_t>(route.expert_id) * rows + row) * k_blocks + block;
        float partial = 0.0f;
        if (!direct_dot_block(expert_wires[wire_index], activation[block], partial)) {
            valid = false;
            break;
        }
        total += partial;
    }
    route_ids[index] = CANDIDATE_UE4M4FAST_PRIVATE;
    expert_ids[index] = route.expert_id;
    if (!valid) {
        atomicAdd(&counters->invalid_count, 1u);
        output[index] = 0.0f;
        return;
    }
    output[index] = total;
}

template <typename T>
struct DeviceArray {
    T * ptr = nullptr;
    size_t count = 0;
    explicit DeviceArray(size_t n) : count(n) { HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T))); }
    ~DeviceArray() { if (ptr) (void) hipFree(ptr); }
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray & operator=(const DeviceArray &) = delete;
};

struct EventPair {
    hipEvent_t start{};
    hipEvent_t stop{};
    EventPair() { HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop)); }
    ~EventPair() { (void) hipEventDestroy(start); (void) hipEventDestroy(stop); }
};

static float f32_from_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t f32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool cpu_decode_weight(const UE4M4FastWire & wire, int weight, float & value) {
    if (weight < 0 || weight >= 32) return false;
    const uint8_t scale_metadata = wire.bytes[16];
    const auto scale = [](uint8_t raw) {
        const int exponent = raw >> 4;
        const int mantissa = raw & 15;
        return exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -11)
                             : std::ldexp(static_cast<float>(16 + mantissa), exponent - 12);
    };
    const uint8_t packed = wire.bytes[weight >> 1];
    const uint8_t code = ((weight & 1) ? (packed >> 4) : packed) & 15u;
    const float s = scale(scale_metadata);
    value = s == 0.0f ? 0.0f : s * static_cast<float>(UE4M4FAST_CODEBOOK[code]);
    return true;
}

static float raw_activation_value(int col, int block, int weight, bool zero) {
    if (zero) return 0.0f;
    const int centered = ((5 * weight + 7 * col + 3 * block + 11) % 23) - 11;
    return 0.01953125f + 0.078125f * static_cast<float>(centered) + 0.00031f * static_cast<float>((weight + col + block) % 13);
}

static Q8Block make_q8_block(int col, int block, bool mmvq, bool zero = false) {
    std::array<float, 32> raw{};
    float amax = 0.0f;
    for (int weight = 0; weight < 32; ++weight) {
        raw[weight] = raw_activation_value(col, block, weight, zero);
        amax = std::max(amax, std::fabs(raw[weight]));
    }
    const float d_unrounded = amax / 127.0f;
    const float inverse = d_unrounded != 0.0f ? 1.0f / d_unrounded : 0.0f;
    Q8Block out{};
    out.d = mmvq ? __half2float(__float2half(d_unrounded)) : d_unrounded;
    for (int weight = 0; weight < 32; ++weight) {
        const int q = static_cast<int>(std::round(raw[weight] * inverse));
        out.qs[weight] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
    }
    return out;
}

static std::vector<UE4M4FastWire> make_wires(int rows, int k_blocks, int seed = 0) {
    constexpr int performance_fixtures = UE4M4FAST_PERFORMANCE_FIXTURE_COUNT;
    static_assert(sizeof(ue4m4fast_device_oracle_cases) / sizeof(ue4m4fast_device_oracle_cases[0]) == UE4M4FAST_VALID_FIXTURE_COUNT);
    std::vector<UE4M4FastWire> wires(static_cast<size_t>(rows) * k_blocks);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < k_blocks; ++block) {
            const int fixture = (row * 131 + block * 17 + seed) % performance_fixtures;
            std::memcpy(wires[static_cast<size_t>(row) * k_blocks + block].bytes,
                        ue4m4fast_device_oracle_cases[fixture].wire, sizeof(UE4M4FastWire));
        }
    }
    return wires;
}

static std::vector<Q8Block> make_activations(int n, int k_blocks, bool mmvq, bool zero = false) {
    std::vector<Q8Block> activations(static_cast<size_t>(n) * k_blocks);
    for (int col = 0; col < n; ++col) {
        for (int block = 0; block < k_blocks; ++block) {
            activations[static_cast<size_t>(col) * k_blocks + block] = make_q8_block(col, block, mmvq, zero);
        }
    }
    return activations;
}

static float cpu_expected_q8(
        const std::vector<UE4M4FastWire> & wires, const std::vector<Q8Block> & activations,
        int row, int col, int k_blocks, bool & valid) {
    valid = true;
    float total = 0.0f;
    for (int block = 0; block < k_blocks; ++block) {
        const UE4M4FastWire & wire = wires[static_cast<size_t>(row) * k_blocks + block];
        const Q8Block & activation = activations[static_cast<size_t>(col) * k_blocks + block];
        float block_scale = 0.0f;
        if (!decode_scale(wire.bytes[16], block_scale)) { valid = false; return 0.0f; }
        int block_sum = 0;
        for (int weight = 0; weight < 32; ++weight) {
            const uint8_t packed = wire.bytes[weight >> 1];
            const uint8_t code = ((weight & 1) ? (packed >> 4) : packed) & 15u;
            block_sum += static_cast<int>(UE4M4FAST_CODEBOOK[code]) * static_cast<int>(activation.qs[weight]);
        }
        total += activation.d * block_scale * static_cast<float>(block_sum);
    }
    return total;
}

static float cpu_expected_raw(
        const std::vector<UE4M4FastWire> & wires, int row, int col, int k_blocks, bool & valid) {
    valid = true;
    float total = 0.0f;
    for (int block = 0; block < k_blocks; ++block) {
        const UE4M4FastWire & wire = wires[static_cast<size_t>(row) * k_blocks + block];
        for (int weight = 0; weight < 32; ++weight) {
            float decoded = 0.0f;
            if (!cpu_decode_weight(wire, weight, decoded)) { valid = false; return 0.0f; }
            total += decoded * raw_activation_value(col, block, weight, false);
        }
    }
    return total;
}

static size_t fixture_mismatches() {
    constexpr int fixtures = sizeof(ue4m4fast_device_oracle_cases) / sizeof(ue4m4fast_device_oracle_cases[0]);
    static_assert(fixtures == UE4M4FAST_VALID_FIXTURE_COUNT);
    size_t mismatches = 0;
    for (int fixture = 0; fixture < fixtures; ++fixture) {
        UE4M4FastWire wire{};
        std::memcpy(wire.bytes, ue4m4fast_device_oracle_cases[fixture].wire, sizeof(UE4M4FastWire));
        for (int weight = 0; weight < 32; ++weight) {
            float value = 0.0f;
            if (!cpu_decode_weight(wire, weight, value) || f32_bits(value) != ue4m4fast_device_oracle_cases[fixture].reconstruction_bits[weight]) ++mismatches;
        }
    }
    return mismatches;
}

struct ShapeResult {
    std::string board;
    std::string role;
    int k = 0;
    int rows = 0;
    int n = 0;
    int k_blocks = 0;
    int warm_repetitions = 0;
    float cold_ms = 0.0f;
    std::vector<float> warm_samples_ms;
    float warm_median_ms = 0.0f;
    float warm_p95_ms = 0.0f;
    DeviceCounters counters{};
    uint64_t expected_dispatch = 0;
    size_t sampled_outputs_checked = 0;
    size_t correctness_mismatches = 0;
    size_t route_mismatches = 0;
    size_t route_discriminants = 0;
    size_t nonfinite_outputs = 0;
    float max_abs_error = 0.0f;
    uint64_t unique_input_bytes = 0;
    uint64_t logical_weights_per_launch = 0;
    uint64_t logical_packed_weight_bytes_per_launch = 0;
    double logical_weights_per_second = 0.0;
    double logical_packed_weight_bytes_per_second = 0.0;
    double outputs_per_second = 0.0;
    bool pass = false;
};

static float percentile(std::vector<float> values, double q) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1, static_cast<size_t>(std::ceil(q * values.size()) - 1));
    return values[index];
}

static void launch_shape_kernel(
        const std::string & board, dim3 grid, DeviceArray<UE4M4FastWire> & wires, DeviceArray<Q8Block> & activations,
        DeviceArray<float> & output, DeviceArray<uint32_t> & routes, DeviceArray<DeviceCounters> & counters,
        int rows, int n, int k_blocks) {
    if (board == "mmvq_decode") {
        ue4m4fast_deepseek_mmvq_kernel<<<grid, dim3(THREADS)>>>(wires.ptr, activations.ptr, output.ptr, routes.ptr, counters.ptr, rows, k_blocks);
    } else {
        ue4m4fast_deepseek_mmq_kernel<<<grid, dim3(THREADS)>>>(wires.ptr, activations.ptr, output.ptr, routes.ptr, counters.ptr, rows, n, k_blocks);
    }
    HIP_CHECK(hipGetLastError());
}

static ShapeResult run_shape(
        const std::string & board, const std::string & role, int k, int rows, int n, int warm_repetitions) {
    ShapeResult result{};
    result.board = board; result.role = role; result.k = k; result.rows = rows; result.n = n;
    const int k_blocks = k / 32;
    result.k_blocks = k_blocks;
    result.warm_repetitions = warm_repetitions;
    const bool mmvq = board == "mmvq_decode";
    std::fprintf(stderr, "shape-start board=%s role=%s K=%d rows=%d N=%d\n", board.c_str(), role.c_str(), k, rows, n);

    const std::vector<UE4M4FastWire> host_wires = make_wires(rows, k_blocks);
    const std::vector<Q8Block> host_activations = make_activations(n, k_blocks, mmvq);
    const size_t output_count = static_cast<size_t>(rows) * n;
    DeviceArray<UE4M4FastWire> device_wires(host_wires.size());
    DeviceArray<Q8Block> device_activations(host_activations.size());
    DeviceArray<float> device_output(output_count);
    DeviceArray<uint32_t> device_routes(output_count);
    DeviceArray<DeviceCounters> device_counters(1);
    HIP_CHECK(hipMemcpy(device_wires.ptr, host_wires.data(), host_wires.size() * sizeof(UE4M4FastWire), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_activations.ptr, host_activations.data(), host_activations.size() * sizeof(Q8Block), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_routes.ptr, 0, output_count * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(device_counters.ptr, 0, sizeof(DeviceCounters)));

    const dim3 grid((rows + THREADS - 1) / THREADS, mmvq ? 1 : n);
    EventPair events;
    HIP_CHECK(hipEventRecord(events.start));
    launch_shape_kernel(board, grid, device_wires, device_activations, device_output, device_routes, device_counters, rows, n, k_blocks);
    HIP_CHECK(hipEventRecord(events.stop));
    HIP_CHECK(hipEventSynchronize(events.stop));
    HIP_CHECK(hipEventElapsedTime(&result.cold_ms, events.start, events.stop));
    for (int repetition = 0; repetition < warm_repetitions; ++repetition) {
        HIP_CHECK(hipEventRecord(events.start));
        launch_shape_kernel(board, grid, device_wires, device_activations, device_output, device_routes, device_counters, rows, n, k_blocks);
        HIP_CHECK(hipEventRecord(events.stop));
        HIP_CHECK(hipEventSynchronize(events.stop));
        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, events.start, events.stop));
        result.warm_samples_ms.push_back(elapsed);
    }
    result.warm_median_ms = percentile(result.warm_samples_ms, 0.5);
    result.warm_p95_ms = percentile(result.warm_samples_ms, 0.95);

    std::vector<float> host_output(output_count);
    std::vector<uint32_t> host_routes(output_count);
    HIP_CHECK(hipMemcpy(host_output.data(), device_output.ptr, host_output.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_routes.data(), device_routes.ptr, host_routes.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&result.counters, device_counters.ptr, sizeof(DeviceCounters), hipMemcpyDeviceToHost));
    for (size_t index = 0; index < output_count; ++index) {
        if (!std::isfinite(host_output[index])) ++result.nonfinite_outputs;
        if (host_routes[index] != CANDIDATE_UE4M4FAST_PRIVATE) ++result.route_mismatches;
    }

    std::vector<int> sample_rows = {0, 1, std::min(127, rows - 1), rows - 1};
    std::vector<int> sample_cols = {0, n / 2, n - 1};
    std::sort(sample_rows.begin(), sample_rows.end()); sample_rows.erase(std::unique(sample_rows.begin(), sample_rows.end()), sample_rows.end());
    std::sort(sample_cols.begin(), sample_cols.end()); sample_cols.erase(std::unique(sample_cols.begin(), sample_cols.end()), sample_cols.end());
    for (int col : sample_cols) {
        for (int row : sample_rows) {
            bool q8_valid = false, raw_valid = false;
            const float expected = cpu_expected_q8(host_wires, host_activations, row, col, k_blocks, q8_valid);
            const float raw_expected = cpu_expected_raw(host_wires, row, col, k_blocks, raw_valid);
            const float actual = host_output[static_cast<size_t>(col) * rows + row];
            const float tolerance = 2.0e-4f + 2.0e-4f * std::fabs(expected);
            ++result.sampled_outputs_checked;
            result.max_abs_error = std::max(result.max_abs_error, std::fabs(actual - expected));
            if (!q8_valid || !raw_valid || !std::isfinite(actual) || std::fabs(actual - expected) > tolerance) ++result.correctness_mismatches;
            if (q8_valid && raw_valid && std::fabs(raw_expected - expected) > 4.0f * tolerance) ++result.route_discriminants;
        }
    }

    result.expected_dispatch = static_cast<uint64_t>(output_count) * static_cast<uint64_t>(warm_repetitions + 1);
    result.unique_input_bytes = static_cast<uint64_t>(host_wires.size() * sizeof(UE4M4FastWire) + host_activations.size() * sizeof(Q8Block));
    result.logical_weights_per_launch = static_cast<uint64_t>(rows) * n * k;
    result.logical_packed_weight_bytes_per_launch = static_cast<uint64_t>(output_count) * k_blocks * sizeof(UE4M4FastWire);
    const double seconds = static_cast<double>(result.warm_median_ms) / 1000.0;
    result.logical_weights_per_second = static_cast<double>(result.logical_weights_per_launch) / seconds;
    result.logical_packed_weight_bytes_per_second = static_cast<double>(result.logical_packed_weight_bytes_per_launch) / seconds;
    result.outputs_per_second = static_cast<double>(output_count) / seconds;
    result.pass = result.cold_ms > 0.0f && result.warm_samples_ms.size() == static_cast<size_t>(warm_repetitions) &&
        result.counters.dispatch_count == result.expected_dispatch && result.counters.invalid_count == 0 && result.counters.fallback_count == 0 &&
        result.sampled_outputs_checked > 0 && result.correctness_mismatches == 0 && result.route_mismatches == 0 &&
        result.route_discriminants > 0 && result.nonfinite_outputs == 0;
    std::fprintf(stderr, "shape-end board=%s role=%s N=%d status=%s median_ms=%.6f\n", board.c_str(), role.c_str(), n, result.pass ? "PASS" : "FAIL", result.warm_median_ms);
    return result;
}

struct GuardResult {
    uint32_t dispatch_count = 0;
    uint32_t invalid_count = 0;
    uint32_t fallback_count = 0;
    size_t wire_domain_checks = 0;
    size_t invalid_wire_states = 0;
    size_t upper_boundary_checks = 0;
    size_t zero_checks = 0;
    size_t tail_checks = 0;
    bool pass = false;
};

static GuardResult run_wire_domain_tail_zero_suite() {
    constexpr int rows = 259;
    constexpr int k_blocks = 1;
    std::vector<UE4M4FastWire> wires = make_wires(rows, k_blocks);
    for (int raw = 0; raw < UE4M4_LEGAL_METADATA_BYTES; ++raw) {
        wires[raw].bytes[16] = static_cast<uint8_t>(raw);
    }
    std::vector<Q8Block> activation = make_activations(1, k_blocks, true);
    DeviceArray<UE4M4FastWire> dw(wires.size()); DeviceArray<Q8Block> da(activation.size()); DeviceArray<float> dout(rows);
    DeviceArray<uint32_t> droutes(rows); DeviceArray<DeviceCounters> dc(1);
    HIP_CHECK(hipMemcpy(dw.ptr, wires.data(), wires.size() * sizeof(UE4M4FastWire), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(da.ptr, activation.data(), activation.size() * sizeof(Q8Block), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(droutes.ptr, 0, rows * sizeof(uint32_t))); HIP_CHECK(hipMemset(dc.ptr, 0, sizeof(DeviceCounters)));
    ue4m4fast_deepseek_mmvq_kernel<<<dim3((rows + THREADS - 1) / THREADS), dim3(THREADS)>>>(dw.ptr, da.ptr, dout.ptr, droutes.ptr, dc.ptr, rows, k_blocks);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> output(rows); std::vector<uint32_t> routes(rows); DeviceCounters counters{};
    HIP_CHECK(hipMemcpy(output.data(), dout.ptr, rows * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(routes.data(), droutes.ptr, rows * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&counters, dc.ptr, sizeof(counters), hipMemcpyDeviceToHost));
    GuardResult result{};
    result.dispatch_count = counters.dispatch_count; result.invalid_count = counters.invalid_count; result.fallback_count = counters.fallback_count;
    for (int raw = 0; raw < UE4M4_LEGAL_METADATA_BYTES; ++raw) {
        bool expected_valid = false;
        const float expected = cpu_expected_q8(wires, activation, raw, 0, k_blocks, expected_valid);
        const float tolerance = 2.0e-4f + 2.0e-4f * std::fabs(expected);
        if (expected_valid && std::isfinite(output[raw]) && routes[raw] == CANDIDATE_UE4M4FAST_PRIVATE && std::fabs(output[raw] - expected) <= tolerance) {
            ++result.wire_domain_checks;
        }
    }
    result.invalid_wire_states = UE4M4_INVALID_METADATA_BYTES;
    if (std::isfinite(output[255]) && routes[255] == CANDIDATE_UE4M4FAST_PRIVATE) ++result.upper_boundary_checks;
    if (counters.dispatch_count == rows) ++result.tail_checks;

    const std::vector<Q8Block> zero_activation = make_activations(1, k_blocks, true, true);
    HIP_CHECK(hipMemcpy(da.ptr, zero_activation.data(), zero_activation.size() * sizeof(Q8Block), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dc.ptr, 0, sizeof(DeviceCounters)));
    ue4m4fast_deepseek_mmvq_kernel<<<dim3((rows + THREADS - 1) / THREADS), dim3(THREADS)>>>(dw.ptr, da.ptr, dout.ptr, droutes.ptr, dc.ptr, rows, k_blocks);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(output.data(), dout.ptr, rows * sizeof(float), hipMemcpyDeviceToHost));
    for (float value : output) if (value == 0.0f) ++result.zero_checks;
    result.pass = result.dispatch_count == rows && result.invalid_count == 0 && result.fallback_count == 0 &&
        result.wire_domain_checks == UE4M4_LEGAL_METADATA_BYTES && result.invalid_wire_states == UE4M4_INVALID_METADATA_BYTES &&
        result.upper_boundary_checks == 1 && result.tail_checks == 1 && result.zero_checks == rows;
    return result;
}

struct RoutedResult {
    uint32_t dispatch_count = 0;
    uint32_t invalid_count = 0;
    uint32_t fallback_count = 0;
    uint32_t inactive_slots = 0;
    size_t duplicate_expert_checks = 0;
    size_t empty_expert_checks = 0;
    size_t malformed_route_checks = 0;
    bool pass = false;
};

static RoutedResult run_duplicate_empty_malformed_expert_suite() {
    constexpr int rows = 33, k_blocks = 2, experts = 2, slots_count = 5;
    const RouteSlot slots[slots_count] = {{0, 1}, {0, 1}, {1, 1}, {-1, 0}, {experts, 1}};
    std::vector<UE4M4FastWire> expert_wires(static_cast<size_t>(experts) * rows * k_blocks);
    for (int expert = 0; expert < experts; ++expert) {
        const std::vector<UE4M4FastWire> one = make_wires(rows, k_blocks, expert * 9);
        std::copy(one.begin(), one.end(), expert_wires.begin() + static_cast<size_t>(expert) * rows * k_blocks);
    }
    const std::vector<Q8Block> activation = make_activations(1, k_blocks, true);
    const size_t outputs = static_cast<size_t>(slots_count) * rows;
    std::vector<float> initial_output(outputs, 12345.0f);
    std::vector<uint32_t> initial_routes(outputs, 0u);
    std::vector<int32_t> initial_experts(outputs, -99);
    DeviceArray<UE4M4FastWire> dw(expert_wires.size()); DeviceArray<Q8Block> da(activation.size()); DeviceArray<RouteSlot> ds(slots_count);
    DeviceArray<float> dout(outputs); DeviceArray<uint32_t> droutes(outputs); DeviceArray<int32_t> dexperts(outputs); DeviceArray<DeviceCounters> dc(1);
    HIP_CHECK(hipMemcpy(dw.ptr, expert_wires.data(), expert_wires.size() * sizeof(UE4M4FastWire), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(da.ptr, activation.data(), activation.size() * sizeof(Q8Block), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ds.ptr, slots, sizeof(slots), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dout.ptr, initial_output.data(), outputs * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(droutes.ptr, initial_routes.data(), outputs * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dexperts.ptr, initial_experts.data(), outputs * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dc.ptr, 0, sizeof(DeviceCounters)));
    ue4m4fast_routed_semantics_kernel<<<dim3((rows + THREADS - 1) / THREADS, slots_count), dim3(THREADS)>>>(
        dw.ptr, da.ptr, ds.ptr, dout.ptr, droutes.ptr, dexperts.ptr, dc.ptr, rows, slots_count, experts, k_blocks);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> output(outputs); std::vector<uint32_t> routes(outputs); std::vector<int32_t> expert_ids(outputs); DeviceCounters counters{};
    HIP_CHECK(hipMemcpy(output.data(), dout.ptr, outputs * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(routes.data(), droutes.ptr, outputs * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(expert_ids.data(), dexperts.ptr, outputs * sizeof(int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&counters, dc.ptr, sizeof(counters), hipMemcpyDeviceToHost));
    RoutedResult result{};
    result.dispatch_count = counters.dispatch_count; result.invalid_count = counters.invalid_count;
    result.fallback_count = counters.fallback_count; result.inactive_slots = counters.inactive_slots;
    for (int row = 0; row < rows; ++row) {
        if (output[row] == output[rows + row] && routes[row] == CANDIDATE_UE4M4FAST_PRIVATE && routes[rows + row] == CANDIDATE_UE4M4FAST_PRIVATE && expert_ids[row] == 0 && expert_ids[rows + row] == 0) ++result.duplicate_expert_checks;
        const size_t empty = static_cast<size_t>(3) * rows + row;
        if (output[empty] == 12345.0f && routes[empty] == 0u && expert_ids[empty] == -99) ++result.empty_expert_checks;
        const size_t malformed = static_cast<size_t>(4) * rows + row;
        if (output[malformed] == 0.0f && routes[malformed] == 0u && expert_ids[malformed] == -99) ++result.malformed_route_checks;
    }
    result.pass = result.dispatch_count == 4u * rows && result.invalid_count == rows && result.fallback_count == 0 &&
        result.inactive_slots == 1 && result.duplicate_expert_checks == rows && result.empty_expert_checks == rows &&
        result.malformed_route_checks == rows;
    return result;
}

static std::string shape_json(const ShapeResult & r) {
    std::ostringstream out; out << std::setprecision(17);
    out << "{\"board\":\"" << r.board << "\",\"role\":\"" << r.role << "\",\"K\":" << r.k
        << ",\"rows\":" << r.rows << ",\"N\":" << r.n << ",\"k_blocks\":" << r.k_blocks
        << ",\"cold_ms\":" << r.cold_ms << ",\"warm_repetitions\":" << r.warm_repetitions << ",\"warm_samples_ms\":[";
    for (size_t i = 0; i < r.warm_samples_ms.size(); ++i) { if (i) out << ','; out << r.warm_samples_ms[i]; }
    out << "],\"warm_median_ms\":" << r.warm_median_ms << ",\"warm_p95_ms\":" << r.warm_p95_ms
        << ",\"dispatch_count\":" << r.counters.dispatch_count << ",\"expected_dispatch\":" << r.expected_dispatch
        << ",\"invalid_count\":" << r.counters.invalid_count << ",\"fallback_count\":" << r.counters.fallback_count
        << ",\"sampled_outputs_checked\":" << r.sampled_outputs_checked << ",\"correctness_mismatches\":" << r.correctness_mismatches
        << ",\"route_mismatches\":" << r.route_mismatches << ",\"route_discriminants\":" << r.route_discriminants
        << ",\"nonfinite_outputs\":" << r.nonfinite_outputs << ",\"max_abs_error\":" << r.max_abs_error
        << ",\"unique_input_bytes\":" << r.unique_input_bytes << ",\"logical_weights_per_launch\":" << r.logical_weights_per_launch
        << ",\"logical_packed_weight_bytes_per_launch\":" << r.logical_packed_weight_bytes_per_launch
        << ",\"logical_weights_per_second\":" << r.logical_weights_per_second
        << ",\"logical_packed_weight_bytes_per_second\":" << r.logical_packed_weight_bytes_per_second
        << ",\"outputs_per_second\":" << r.outputs_per_second << ",\"status\":\"" << (r.pass ? "PASS" : "FAIL") << "\"}";
    return out.str();
}

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    if (std::strcmp(prop.gcnArchName, "gfx1151") != 0 || prop.warpSize != 32) {
        std::fprintf(stderr, "unexpected device arch=%s wave=%d\n", prop.gcnArchName, prop.warpSize);
        return 1;
    }
    const int warm_repetitions = []() {
        const char * value = std::getenv("UE4M4FAST_WARM_REPETITIONS");
        const int parsed = value ? std::atoi(value) : 5;
        return parsed >= 5 && parsed <= 15 ? parsed : 5;
    }();
    const size_t fixtures_bad = fixture_mismatches();
    const GuardResult guards = run_wire_domain_tail_zero_suite();
    const RoutedResult routed = run_duplicate_empty_malformed_expert_suite();
    std::vector<ShapeResult> shapes;
    shapes.push_back(run_shape("mmvq_decode", "gate_up", GATE_K, GATE_ROWS, 1, warm_repetitions));
    shapes.push_back(run_shape("mmvq_decode", "down", DOWN_K, DOWN_ROWS, 1, warm_repetitions));
    for (int n : MMQ_NS) shapes.push_back(run_shape("mmq_prefill", "gate_up", GATE_K, GATE_ROWS, n, warm_repetitions));
    for (int n : MMQ_NS) shapes.push_back(run_shape("mmq_prefill", "down", DOWN_K, DOWN_ROWS, n, warm_repetitions));

    hipFuncAttributes mmvq_attr{}, mmq_attr{}, routed_attr{};
    HIP_CHECK(hipFuncGetAttributes(&mmvq_attr, reinterpret_cast<const void *>(ue4m4fast_deepseek_mmvq_kernel)));
    HIP_CHECK(hipFuncGetAttributes(&mmq_attr, reinterpret_cast<const void *>(ue4m4fast_deepseek_mmq_kernel)));
    HIP_CHECK(hipFuncGetAttributes(&routed_attr, reinterpret_cast<const void *>(ue4m4fast_routed_semantics_kernel)));
    int mmvq_blocks_per_cu = 0, mmq_blocks_per_cu = 0, routed_blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&mmvq_blocks_per_cu, ue4m4fast_deepseek_mmvq_kernel, THREADS, 0));
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&mmq_blocks_per_cu, ue4m4fast_deepseek_mmq_kernel, THREADS, 0));
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&routed_blocks_per_cu, ue4m4fast_routed_semantics_kernel, THREADS, 0));
    const int waves_per_block = THREADS / prop.warpSize;
    bool pass = fixtures_bad == 0 && guards.pass && routed.pass;
    for (const ShapeResult & shape : shapes) pass = pass && shape.pass;

    std::ostringstream out; out << std::setprecision(17);
    out << "{\"schema\":\"gfx1151-ue4m4fast-deepseek-runtime-v1\",\"status\":\""
        << (pass ? "RUNTIME_PASS_AWAITING_ISA_PARENT_GATE" : "RUNTIME_FAIL_NOT_BOARD_RESULT")
        << "\",\"counts_as_board_result\":false,\"candidate\":\"rocmfp4-ue4m4-fast\",\"candidate_id\":" << CANDIDATE_UE4M4FAST_PRIVATE
        << ",\"device\":{\"name\":\"" << prop.name << "\",\"gcn_arch\":\"" << prop.gcnArchName << "\",\"wave_size\":" << prop.warpSize << ",\"compute_units\":" << prop.multiProcessorCount << "}"
        << ",\"fixtures\":{\"valid_cases\":" << UE4M4FAST_VALID_FIXTURE_COUNT
        << ",\"source_matched_cases\":" << UE4M4FAST_PERFORMANCE_FIXTURE_COUNT << ",\"synthetic_canary_cases\":0"
        << ",\"fixture_mismatches\":" << fixtures_bad << "}"
        << ",\"guards\":{\"status\":\"" << (guards.pass ? "PASS" : "FAIL") << "\",\"dispatch_count\":" << guards.dispatch_count
        << ",\"invalid_count\":" << guards.invalid_count << ",\"fallback_count\":" << guards.fallback_count
        << ",\"wire_domain_checks\":" << guards.wire_domain_checks << ",\"invalid_wire_states\":" << guards.invalid_wire_states
        << ",\"upper_boundary_checks\":" << guards.upper_boundary_checks << ",\"zero_checks\":" << guards.zero_checks
        << ",\"tail_checks\":" << guards.tail_checks << ",\"malformed_wire_contract\":\"" << UE4M4_MALFORMED_WIRE_CONTRACT << "\"}"
        << ",\"routed_semantics\":{\"status\":\"" << (routed.pass ? "PASS" : "FAIL") << "\",\"dispatch_count\":" << routed.dispatch_count
        << ",\"invalid_count\":" << routed.invalid_count << ",\"fallback_count\":" << routed.fallback_count << ",\"inactive_slots\":" << routed.inactive_slots
        << ",\"duplicate_expert_checks\":" << routed.duplicate_expert_checks << ",\"empty_expert_checks\":" << routed.empty_expert_checks
        << ",\"malformed_route_checks\":" << routed.malformed_route_checks << "}"
        << ",\"shapes\":[";
    for (size_t i = 0; i < shapes.size(); ++i) { if (i) out << ','; out << shape_json(shapes[i]); }
    out << "],\"resources_runtime\":{\"mmvq\":{\"num_regs\":" << mmvq_attr.numRegs << ",\"shared_bytes\":" << mmvq_attr.sharedSizeBytes
        << ",\"local_bytes\":" << mmvq_attr.localSizeBytes << ",\"max_threads\":" << mmvq_attr.maxThreadsPerBlock
        << ",\"blocks_per_compute_unit\":" << mmvq_blocks_per_cu << ",\"active_waves_per_compute_unit\":" << mmvq_blocks_per_cu * waves_per_block
        << "},\"mmq\":{\"num_regs\":" << mmq_attr.numRegs << ",\"shared_bytes\":" << mmq_attr.sharedSizeBytes
        << ",\"local_bytes\":" << mmq_attr.localSizeBytes << ",\"max_threads\":" << mmq_attr.maxThreadsPerBlock
        << ",\"blocks_per_compute_unit\":" << mmq_blocks_per_cu << ",\"active_waves_per_compute_unit\":" << mmq_blocks_per_cu * waves_per_block
        << "},\"routed\":{\"num_regs\":" << routed_attr.numRegs << ",\"shared_bytes\":" << routed_attr.sharedSizeBytes
        << ",\"local_bytes\":" << routed_attr.localSizeBytes << ",\"max_threads\":" << routed_attr.maxThreadsPerBlock
        << ",\"blocks_per_compute_unit\":" << routed_blocks_per_cu << ",\"active_waves_per_compute_unit\":" << routed_blocks_per_cu * waves_per_block << "}}}"
        << '\n';
    std::printf("%s", out.str().c_str());
    return pass ? 0 : 1;
}
