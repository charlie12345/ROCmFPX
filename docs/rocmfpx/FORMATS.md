# Format and compatibility contract

ROCmFP2/3/4/6/8 are immutable disk formats. Optimizations may change scale
search, dispatch, vectorization, shader organization, or cached repacks, but
must not change block bytes, element order, scale encoding, block size, GGML
type IDs, or dequantized meaning. Golden byte vectors and CPU dequantization are
the authority for backend tests.

ROCmFP5 and ROCmFP7 version 1 use 32 weights, two unsigned finite UE4M3 scale
bytes (one per 16 weights), and dense little-endian signed-magnitude payloads.
FP5 stores 5-bit codes in 20 bytes plus two scales (22 bytes, 5.50 bpw). FP7
stores 7-bit codes in 28 bytes plus two scales (30 bytes, 7.50 bpw). A zero
magnitude ignores the sign bit. Both quantizers perform an exhaustive
reconstruction-error scale search and accept importance weights.

The `NVFP4` preset is the native-retention path. If a source tensor is already
NVFP4 and its selected destination is NVFP4, the quantizer's same-type path
copies its encoded bytes; no ROCmFP4 conversion is required. Converting NVFP4
to ROCmFP4 remains explicitly available but is not bit-exact because the top
codebook value differs.

Any future incompatible layout needs a new GGML type ID and a documented
format version. It must not reuse an existing ROCmFPX name.
