# Format and compatibility contract

ROCmFP2/3/4/6/8 are immutable disk formats. Optimizations may change scale
search, dispatch, vectorization, shader organization, or cached repacks, but
must not change block bytes, element order, scale encoding, block size, GGML
type IDs, or dequantized meaning. Golden byte vectors and CPU dequantization are
the authority for backend tests.

Every block holds 32 weights. The type names are the ones `llama-quantize`
and `--tensor-type` take:

| Format | Type name | GGML type | Block bytes | bpw |
| --- | --- | ---: | --- | ---: |
| ROCmFP2 | `Q2_0_ROCMFPX` | 111 | 8 codes + 2 scales = 10 | 2.50 |
| ROCmFP3 | `Q3_0_ROCMFPX` | 104 | 12 + 2 = 14 | 3.50 |
| ROCmFP4 | `Q4_0_ROCMFP4` | 100 | 16 + 2 = 18 | 4.50 |
| ROCmFP4 fast | `Q4_0_ROCMFP4_FAST` | 101 | 16 + 1 = 17 | 4.25 |
| ROCmI4 | `Q4_0_ROCMI4` | 108 | 16 + 1 = 17 | 4.25 |
| ROCmFP5 | `Q5_0_ROCMFPX` | 109 | 20 + 2 = 22 | 5.50 |
| ROCmFP6 | `Q6_0_ROCMFPX` | 102 | 24 + 2 = 26 | 6.50 |
| ROCmFP7 | `Q7_0_ROCMFPX` | 110 | 28 + 2 = 30 | 7.50 |
| ROCmFP8 | `Q8_0_ROCMFPX` | 103 | 32 + 1 = 33 | 8.25 |

Types 105 and 106 are the TurboQuant KV-cache types, not ROCmFPX weight
formats. ROCmFP2's tensor type moved to 111 with the same 10-byte payload:
type 107 was written with two incompatible ROCmFP2 layouts, so readers reject
it. [ROCmFP2 tensor type identity](../../README.md#rocmfp2-tensor-type-identity)
has the audit and retag tool for older files.

ROCmFP5 and ROCmFP7 version 1 use 32 weights, two unsigned finite UE4M3 scale
bytes (one per 16 weights), and dense little-endian signed-magnitude payloads.
FP5 stores 5-bit codes in 20 bytes plus two scales (22 bytes, 5.50 bpw). FP7
stores 7-bit codes in 28 bytes plus two scales (30 bytes, 7.50 bpw). A zero
magnitude ignores the sign bit. Both quantizers perform an exhaustive
reconstruction-error scale search and accept importance weights.

ROCmI4 is not in the frozen list. Its block is 32 two's-complement 4-bit
integers and one UE4M3 scale: the size of `Q4_0_ROCMFP4_FAST`, but the codes
are the integer values, with no codebook. The opt-in W4A4 build is described in
[ROCMI4.md](../../ggml/rocmfpx/ROCMI4.md).

The `NVFP4` preset is the native-retention path. If a source tensor is already
NVFP4 and its selected destination is NVFP4, the quantizer's same-type path
copies its encoded bytes; no ROCmFP4 conversion is required. Converting NVFP4
to ROCmFP4 remains explicitly available but is not bit-exact because the top
codebook value differs.

Any future incompatible layout needs a new GGML type ID and a documented
format version. It must not reuse an existing ROCmFPX name.
