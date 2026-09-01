#include "llama-mmap.h"

#undef NDEBUG
#include <cassert>

int main() {
    const llama_mmap::ranges lazy = {{100, 200}, {400, 500}};
    const auto ranges = llama_mmap::initial_prefetch_ranges(600, lazy, 600);

    assert(ranges.size() == 3);
    const std::pair<size_t, size_t> expected0{0, 100};
    const std::pair<size_t, size_t> expected1{200, 400};
    const std::pair<size_t, size_t> expected2{500, 600};
    assert(ranges[0] == expected0);
    assert(ranges[1] == expected1);
    assert(ranges[2] == expected2);

    return 0;
}
