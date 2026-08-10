# UE4M4 experiment provenance and redistribution boundary

This record covers the exact two-file compile bundle in this directory. It contains no local filesystem paths.

## Published-byte candidates

| File | SHA-256 |
|---|---|
| `ue4m4fast_deepseek_board.hip` | `b88c59cce0a2950e99bd9849dc3a3bd0e1de81dfc6d31016569bce45bd79ec53` |
| `ue4m4fast-source-matched-device-oracle-v1.inc` | `66824df16ca1c07950cdfe42153b941ffe8a4fee0d0ec6eb980fb369220fedb9` |

The HIP file includes the `.inc` file by relative quoted include. Both are required for compilation.

## Source derivation

The HIP source was rendered mechanically from a sealed local standalone parent with SHA-256 `e911d1058d47f39e395e412633df138a6f10ec22c3e51049932c8c27487c93c6`. The renderer checked that parent identity before applying candidate renaming, the unsigned UE4M4 scale contract, complete metadata-domain guards, and routed-semantics checks. Re-rendering produced the exact HIP hash above.

The public ROCmFPX base used for this review is `charlie12345/ROCmFPX` `official/main` at commit `00d54526e24e3aba4c76474e3147cbf9c7cc034c`, whose repository license is MIT.

## Fixture derivation

The `.inc` contains 64 deterministic oracle vectors. Each vector records 32 source float bit patterns, one 17-byte packed Codebook10/UE4M4 wire, and 32 expected reconstruction bit patterns. The source names are `source-structured-*` and `source-random-*`; they are bounded test vectors, not model or checkpoint tensors.

Recorded generation-input identities:

| Input | SHA-256 |
|---|---|
| shared low-bit oracle | `5d3064c5ef242ae9a0dbe8dd6bfb4e5c6542cdcc30a28dd65e97f83dd376060d` |
| FP4 oracle | `343dff5736a75d914282918d2b70121369233c9c0026d9a8584d48b687c6d6a5` |
| deterministic S40 fixture | `53219ccc12d64e27eed80af6d65caf1afe5c4c224f49054f94ad67cb281f5971` |
| corrected-oracle adapter | `01699a32bd90f238db27421110abb9cb357980081917c03a84ef49b981367c66` |
| corrected-oracle lock | `380f1050fa1a4489793f38c3f768618dd87faa18174d3a786d71f477a6952a65` |
| successor-surveillance adapter | `904b8f7eabcabaee251fcde2e26bbaea8c9714033a2722367fab230d7d163be2` |
| successor-surveillance lock | `a32f56b038214603fb417d1f5cc6a744ae9b0925db67aa97bd059fecda1fd608` |
| donor implementation source | `939981f37117793c3e75bda7610cb8f85128bf7790790aa3833a8a795a7902ea` |

The original S40 test vectors were generated deterministically from explicit structured cases and a fixed-seed pseudo-random sequence. The UE4M4 fixture then deterministically quantized those accepted reconstructed values using the fixed Codebook10 and unsigned UE4M4 scale grid.

## Redistribution boundary

Available evidence supports reproducible byte identity and shows that the parent repository and donor llama.cpp tree use MIT licensing. It does not establish a complete public Git license chain for every untracked local oracle, adapter, lock, and generated fixture input.

The human contributor has supplied the missing redistribution attestation for the exact staged bytes. The contributor confirms review and understanding of the source, synthetic fixture, provenance, and evidence; authority to distribute the exact HIP source and synthetic fixture under MIT; approval of the parent attribution; preservation of the repository MIT notice; and acceptance of ownership, explanation, and maintenance responsibility.

Therefore the exact two-file bundle identified by the hashes above is cleared for source-stage public publication under MIT. This attestation applies only to those exact bytes and does not publish or license the underlying local fixture report, generator, adapters, or locks as separate artifacts.

Do not publish the original local fixture report or generator unchanged: they contain workstation paths. They are provenance inputs, not compile dependencies.
