#pragma once
// hipCUB shim for the HIP build: lets the ggml-cuda sources that `#include <cub/cub.cuh>`
// (argsort, top-k, cumsum, mean, sum, ssm-scan) compile unchanged against ROCm's
// CUB-compatible hipCUB. Qualified lookups such as cub::DeviceRadixSort resolve through the
// using-directive, and `using namespace cub;` in those files works as on CUDA.
// Only on the include path when GGML_HIP_CUB is enabled (see ggml-hip/CMakeLists.txt).
#include <hipcub/hipcub.hpp>
namespace cub {
using namespace hipcub;
}
