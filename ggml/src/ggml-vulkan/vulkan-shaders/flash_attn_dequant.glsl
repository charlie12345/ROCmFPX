// Asymmetric K/V flash attention: aliased SSBO views of bindings 1 (K) and 2 (V)
// covering every supported FA element type, plus an uber dequantize4() that
// switches on FaTypeK / FaTypeV. After spec-constant specialization the driver
// folds away every path except the one matching the K/V type for this pipeline.
//
// Included by flash_attn.comp and flash_attn_cm1.comp. Not included by
// flash_attn_cm2.comp, which has its own buffer_reference-based decode path.
//
// We use macros (rather than per-quant decode functions taking a struct) on
// purpose: the FA shaders don't enable GL_EXT_shader_explicit_arithmetic_types_float16
// when FLOAT16 isn't defined, which makes float16-containing struct values
// illegal to return from / pass to functions. Macros expand inline where the
// float16 stays in storage and is converted to FLOAT_TYPE at use.

// F32 is fed as a vec4 "block" (4 floats), matching what dequant_funcs_cm2.glsl
// does for F32 in the cm2 shader. FaBlockBytesK/V == 16 for F32.
layout (binding = 1) readonly buffer K_PACKED_F32  { vec4 data[]; }                k_packed_f32;
layout (binding = 2) readonly buffer V_PACKED_F32  { vec4 data[]; }                v_packed_f32;

layout (binding = 1) readonly buffer K_PACKED_Q4_0 { block_q4_0_packed16 data[]; } k_packed_q4_0;
layout (binding = 2) readonly buffer V_PACKED_Q4_0 { block_q4_0_packed16 data[]; } v_packed_q4_0;
layout (binding = 1) readonly buffer K_PACKED_Q4_1 { block_q4_1_packed16 data[]; } k_packed_q4_1;
layout (binding = 2) readonly buffer V_PACKED_Q4_1 { block_q4_1_packed16 data[]; } v_packed_q4_1;
layout (binding = 1) readonly buffer K_PACKED_Q5_0 { block_q5_0_packed16 data[]; } k_packed_q5_0;
layout (binding = 2) readonly buffer V_PACKED_Q5_0 { block_q5_0_packed16 data[]; } v_packed_q5_0;
layout (binding = 1) readonly buffer K_PACKED_Q5_1 { block_q5_1_packed16 data[]; } k_packed_q5_1;
layout (binding = 2) readonly buffer V_PACKED_Q5_1 { block_q5_1_packed16 data[]; } v_packed_q5_1;
layout (binding = 1) readonly buffer K_PACKED_Q8_0 { block_q8_0_packed16 data[]; } k_packed_q8_0;
layout (binding = 2) readonly buffer V_PACKED_Q8_0 { block_q8_0_packed16 data[]; } v_packed_q8_0;
layout (binding = 1) readonly buffer K_PACKED_ROCMFP4 { block_rocmfp4 data[]; } k_packed_rocmfp4;
layout (binding = 2) readonly buffer V_PACKED_ROCMFP4 { block_rocmfp4 data[]; } v_packed_rocmfp4;
layout (binding = 1) readonly buffer K_PACKED_ROCMFP4_FAST { block_rocmfp4_fast data[]; } k_packed_rocmfp4_fast;
layout (binding = 2) readonly buffer V_PACKED_ROCMFP4_FAST { block_rocmfp4_fast data[]; } v_packed_rocmfp4_fast;
layout (binding = 1) readonly buffer K_PACKED_ROCMFPX_FP3 { block_rocmfpx_fp3 data[]; } k_packed_rocmfpx_fp3;
layout (binding = 2) readonly buffer V_PACKED_ROCMFPX_FP3 { block_rocmfpx_fp3 data[]; } v_packed_rocmfpx_fp3;
layout (binding = 1) readonly buffer K_PACKED_ROCMFPX_FP6 { block_rocmfpx_fp6 data[]; } k_packed_rocmfpx_fp6;
layout (binding = 2) readonly buffer V_PACKED_ROCMFPX_FP6 { block_rocmfpx_fp6 data[]; } v_packed_rocmfpx_fp6;
layout (binding = 1) readonly buffer K_PACKED_ROCMFPX_FP8 { block_rocmfpx_fp8 data[]; } k_packed_rocmfpx_fp8;
layout (binding = 2) readonly buffer V_PACKED_ROCMFPX_FP8 { block_rocmfpx_fp8 data[]; } v_packed_rocmfpx_fp8;
layout (binding = 1) readonly buffer K_PACKED_TURBO3_0 { block_turbo3_0 data[]; } k_packed_turbo3_0;
layout (binding = 2) readonly buffer V_PACKED_TURBO3_0 { block_turbo3_0 data[]; } v_packed_turbo3_0;
layout (binding = 1) readonly buffer K_PACKED_TURBO4_0 { block_turbo4_0 data[]; } k_packed_turbo4_0;
layout (binding = 2) readonly buffer V_PACKED_TURBO4_0 { block_turbo4_0 data[]; } v_packed_turbo4_0;
layout (binding = 1) readonly buffer K_PACKED_IQ4_NL { block_iq4_nl_packed16 data[]; } k_packed_iq4_nl;
layout (binding = 2) readonly buffer V_PACKED_IQ4_NL { block_iq4_nl_packed16 data[]; } v_packed_iq4_nl;
layout (binding = 1) readonly buffer K_PACKED_Q1_0 { block_q1_0 data[]; } k_packed_q1_0;
layout (binding = 2) readonly buffer V_PACKED_Q1_0 { block_q1_0 data[]; } v_packed_q1_0;

layout (binding = 1) readonly buffer K_PACKED_BF16 { u16vec4 data[]; } k_packed_bf16;
layout (binding = 2) readonly buffer V_PACKED_BF16 { u16vec4 data[]; } v_packed_bf16;

// Q4_1 and Q5_1 packed32 views: aliased to the same memory as the packed16
// views, used by the MMQ K-side hot path for fast 4-uint loads.
layout (binding = 1) readonly buffer K_PACKED_Q4_1_P32 { block_q4_1_packed32 data[]; } k_packed_q4_1_p32;
layout (binding = 1) readonly buffer K_PACKED_Q5_1_P32 { block_q5_1_packed32 data[]; } k_packed_q5_1_p32;

const int8_t fa_kvalues_rocmfp4_const[16] = {
    int8_t(0), int8_t(1), int8_t(2), int8_t(3), int8_t(4), int8_t(6), int8_t(8), int8_t(10),
    int8_t(0), int8_t(-1), int8_t(-2), int8_t(-3), int8_t(-4), int8_t(-6), int8_t(-8), int8_t(-10),
};

int8_t fa_rocmfp4_code_i8(uint q) {
    return fa_kvalues_rocmfp4_const[q & 0xFu];
}

FLOAT_TYPE fa_rocmfp4_code_value(uint q) {
    return FLOAT_TYPE(fa_rocmfp4_code_i8(q));
}

int32_t fa_rocmfp4_pack4_i8(uint vui) {
    return pack32(i8vec4(fa_rocmfp4_code_i8( vui        & 0xFu),
                         fa_rocmfp4_code_i8((vui >>  8) & 0xFu),
                         fa_rocmfp4_code_i8((vui >> 16) & 0xFu),
                         fa_rocmfp4_code_i8((vui >> 24) & 0xFu)));
}

FLOAT_TYPE fa_rocmfp4_ue4m3_to_fp_half(uint8_t x) {
    const uint u = uint(x);
    if (u == 0u || u == 127u || u == 255u) {
        return FLOAT_TYPE(0.0);
    }

    const uint exp = (u >> 3) & 15u;
    const uint man = u & 7u;
    if (exp == 0u) {
        return FLOAT_TYPE(float(man) * (1.0 / 1024.0));
    }

    const uint bits = (exp + 119u) << 23 | (man << 20);
    return FLOAT_TYPE(uintBitsToFloat(bits));
}

FLOAT_TYPE fa_turbo_pick4(uint q, FLOAT_TYPE v0, FLOAT_TYPE v1, FLOAT_TYPE v2, FLOAT_TYPE v3) {
    const bool b0 = (q & 1u) != 0u;
    const bool b1 = (q & 2u) != 0u;
    const FLOAT_TYPE v01 = b0 ? v1 : v0;
    const FLOAT_TYPE v23 = b0 ? v3 : v2;
    return b1 ? v23 : v01;
}

FLOAT_TYPE fa_turbo_pick8(uint q, FLOAT_TYPE v0, FLOAT_TYPE v1, FLOAT_TYPE v2, FLOAT_TYPE v3,
                                  FLOAT_TYPE v4, FLOAT_TYPE v5, FLOAT_TYPE v6, FLOAT_TYPE v7) {
    const bool b2 = (q & 4u) != 0u;
    const FLOAT_TYPE v03 = fa_turbo_pick4(q, v0, v1, v2, v3);
    const FLOAT_TYPE v47 = fa_turbo_pick4(q, v4, v5, v6, v7);
    return b2 ? v47 : v03;
}

FLOAT_TYPE fa_turbo3_value(uint q) {
#if defined(FLOAT16)
    const uint mag_idx = (q & 3u) ^ (((q & 4u) == 0u) ? 3u : 0u);
    const FLOAT_TYPE mag = fa_turbo_pick4(mag_idx,
        FLOAT_TYPE(0.0216041461), FLOAT_TYPE(0.0665854520),
        FLOAT_TYPE(0.1181396281), FLOAT_TYPE(0.1883970748));
    return ((q & 4u) != 0u) ? mag : -mag;
#else
    return fa_turbo_pick8(q,
        FLOAT_TYPE(-0.1883972972), FLOAT_TYPE(-0.1181399059),
        FLOAT_TYPE(-0.0665857641), FLOAT_TYPE(-0.0216044751),
        FLOAT_TYPE( 0.0216041461), FLOAT_TYPE( 0.0665854520),
        FLOAT_TYPE( 0.1181396281), FLOAT_TYPE( 0.1883970748));
#endif
}

FLOAT_TYPE fa_turbo4_value(uint q) {
    return fa_turbo_pick8(q & 7u,
        ((q & 8u) != 0u) ? FLOAT_TYPE( 0.0112761586) : FLOAT_TYPE(-0.2376389871),
        ((q & 8u) != 0u) ? FLOAT_TYPE( 0.0341139667) : FLOAT_TYPE(-0.1808080141),
        ((q & 8u) != 0u) ? FLOAT_TYPE( 0.0577250301) : FLOAT_TYPE(-0.1417777640),
        ((q & 8u) != 0u) ? FLOAT_TYPE( 0.0827738972) : FLOAT_TYPE(-0.1102646123),
        ((q & 8u) != 0u) ? FLOAT_TYPE( 0.1102295202) : FLOAT_TYPE(-0.0828112376),
        ((q & 8u) != 0u) ? FLOAT_TYPE( 0.1417455465) : FLOAT_TYPE(-0.0577640422),
        ((q & 8u) != 0u) ? FLOAT_TYPE( 0.1807794468) : FLOAT_TYPE(-0.0341540905),
        ((q & 8u) != 0u) ? FLOAT_TYPE( 0.2376153882) : FLOAT_TYPE(-0.0113168380));
}

FLOAT_TYPEV4 fa_turbo3_values(u8vec4 q) {
    return FLOAT_TYPEV4(fa_turbo3_value(uint(q.x)), fa_turbo3_value(uint(q.y)),
                        fa_turbo3_value(uint(q.z)), fa_turbo3_value(uint(q.w)));
}

FLOAT_TYPEV4 fa_turbo4_values(u8vec4 q) {
    return FLOAT_TYPEV4(fa_turbo4_value(uint(q.x)), fa_turbo4_value(uint(q.y)),
                        fa_turbo4_value(uint(q.z)), fa_turbo4_value(uint(q.w)));
}

uint fa_rocmfpx_fp3_get_bits_qs(const uint8_t qs[12], uint bit_pos) {
    uint code = 0u;
    [[unroll]] for (uint bit = 0u; bit < 3u; ++bit) {
        const uint src_bit = bit_pos + bit;
        code |= ((uint(qs[src_bit >> 3u]) >> (src_bit & 7u)) & 1u) << bit;
    }
    return code;
}

int fa_rocmfpx_fp3_decode(uint code) {
    const uint mag_code = code & 3u;
    const int mag = mag_code == 3u ? 4 : int(mag_code);
    return (code & 4u) != 0u ? -mag : mag;
}

int32_t fa_rocmfpx_fp3_pack4_qs(const uint8_t qs[12], uint ei) {
    const uint start_bit = ei * 3u;
    const uint reg_shift = start_bit & 31u;
    const uint reg_idx = start_bit >> 5;
    const uint qs0 = pack32(u8vec4(qs[0], qs[1], qs[2], qs[3]));
    const uint qs1 = pack32(u8vec4(qs[4], qs[5], qs[6], qs[7]));
    const uint qs2 = pack32(u8vec4(qs[8], qs[9], qs[10], qs[11]));
    const uint val_low  = reg_idx == 0u ? qs0 : (reg_idx == 1u ? qs1 : qs2);
    const uint val_high = reg_idx == 0u ? qs1 : (reg_idx == 1u ? qs2 : 0u);
    const uint bits12 = reg_shift == 0u ?
        (val_low & 0xFFFu) :
        (((val_low >> reg_shift) | (val_high << (32u - reg_shift))) & 0xFFFu);
    return pack32(i8vec4(int8_t(fa_rocmfpx_fp3_decode(bits12 & 7u)),
                         int8_t(fa_rocmfpx_fp3_decode((bits12 >> 3) & 7u)),
                         int8_t(fa_rocmfpx_fp3_decode((bits12 >> 6) & 7u)),
                         int8_t(fa_rocmfpx_fp3_decode((bits12 >> 9) & 7u))));
}

const int8_t fa_kvalues_rocmfpx_fp6_const[64] = {
    int8_t(0), int8_t(1), int8_t(2), int8_t(3), int8_t(4), int8_t(5), int8_t(6), int8_t(7),
    int8_t(8), int8_t(9), int8_t(10), int8_t(11), int8_t(12), int8_t(13), int8_t(14), int8_t(15),
    int8_t(16), int8_t(17), int8_t(18), int8_t(19), int8_t(20), int8_t(21), int8_t(22), int8_t(23),
    int8_t(24), int8_t(25), int8_t(26), int8_t(27), int8_t(28), int8_t(29), int8_t(30), int8_t(31),
    int8_t(0), int8_t(-1), int8_t(-2), int8_t(-3), int8_t(-4), int8_t(-5), int8_t(-6), int8_t(-7),
    int8_t(-8), int8_t(-9), int8_t(-10), int8_t(-11), int8_t(-12), int8_t(-13), int8_t(-14), int8_t(-15),
    int8_t(-16), int8_t(-17), int8_t(-18), int8_t(-19), int8_t(-20), int8_t(-21), int8_t(-22), int8_t(-23),
    int8_t(-24), int8_t(-25), int8_t(-26), int8_t(-27), int8_t(-28), int8_t(-29), int8_t(-30), int8_t(-31)
};

uint fa_rocmfpx_fp6_code_at(uint q0, uint q1, uint q2, uint q3, uint q4, uint q5, uint bit_pos) {
    const uint reg_idx = bit_pos >> 5;
    const uint shift = bit_pos & 31u;
    const uint low  = reg_idx == 0u ? q0 : reg_idx == 1u ? q1 : reg_idx == 2u ? q2 :
                      reg_idx == 3u ? q3 : reg_idx == 4u ? q4 : q5;
    const uint high = reg_idx == 0u ? q1 : reg_idx == 1u ? q2 : reg_idx == 2u ? q3 :
                      reg_idx == 3u ? q4 : reg_idx == 4u ? q5 : 0u;
    uint bits = low >> shift;
    if (shift > 26u) {
        bits |= high << (32u - shift);
    }
    return bits & 0x3Fu;
}

int32_t fa_rocmfpx_fp6_pack4_regs(uint qs0, uint qs1, uint qs2, uint qs3, uint qs4, uint qs5, uint ei) {
    const uint b0 = ei * 6u;
    return pack32(i8vec4(
        fa_kvalues_rocmfpx_fp6_const[fa_rocmfpx_fp6_code_at(qs0, qs1, qs2, qs3, qs4, qs5, b0 + 0u)],
        fa_kvalues_rocmfpx_fp6_const[fa_rocmfpx_fp6_code_at(qs0, qs1, qs2, qs3, qs4, qs5, b0 + 6u)],
        fa_kvalues_rocmfpx_fp6_const[fa_rocmfpx_fp6_code_at(qs0, qs1, qs2, qs3, qs4, qs5, b0 + 12u)],
        fa_kvalues_rocmfpx_fp6_const[fa_rocmfpx_fp6_code_at(qs0, qs1, qs2, qs3, qs4, qs5, b0 + 18u)]));
}

int32_t fa_rocmfpx_fp6_pack4_qs(const int8_t qs[32], uint ei) {
    return pack32(i8vec4(qs[ei + 0u], qs[ei + 1u], qs[ei + 2u], qs[ei + 3u]));
}

#if defined(FA_ROCMFPX_FAMILY)
FLOAT_TYPE fa_ue4m3_to_fp(uint8_t x) {
    return FLOAT_TYPE(ue4m3_fp32_lut[min(uint(x), 127u)]);
}
#else
FLOAT_TYPE fa_ue4m3_to_fp(uint8_t x) {
    return fa_rocmfp4_ue4m3_to_fp_half(x);
}
#endif

int32_t fa_rocmfpx_fp8_pack4_qs(const int8_t qs[32], uint ei) {
    return pack32(i8vec4(qs[ei + 0u], qs[ei + 1u], qs[ei + 2u], qs[ei + 3u]));
}

// Per-quant decode bodies are expanded once for the K view set and once for
// the V view set. The macros take the buffer name as a parameter.
#define FA_DEQUANT4_F32(BUF) \
    return FLOAT_TYPEV4(BUF.data[a_offset + ib]);

#define FA_DEQUANT4_Q4_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));                  \
}

#define FA_DEQUANT4_Q4_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * nibbles                                        \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q5_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = uint(BUF.data[a_offset + ib].qh[0])                                                 \
            | (uint(BUF.data[a_offset + ib].qh[1]) << 16);                                        \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));            \
}

#define FA_DEQUANT4_Q5_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = BUF.data[a_offset + ib].qh;                                                         \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb)                                 \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q8_0(BUF) {                                                                   \
    const i8vec2 v0 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2    ])).xy;               \
    const i8vec2 v1 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2 + 1])).xy;               \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);          \
}

#define FA_DEQUANT4_ROCMFP4(BUF) {                                                                \
    const uint qbase = iqs & 0xFu;                                                                \
    uint vui = pack32(u8vec4(BUF.data[a_offset + ib].qs[qbase + 0u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 1u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 2u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 3u]));                            \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui >>= shift;                                                                                \
    const uint half_idx = (iqs & 0x10) != 0 ? 1u : 0u;                                            \
    const FLOAT_TYPE d = fa_rocmfp4_ue4m3_to_fp_half(BUF.data[a_offset + ib].e[half_idx]);        \
    return d * FLOAT_TYPEV4(fa_rocmfp4_code_value( vui        & 0xF),                             \
                            fa_rocmfp4_code_value((vui >>  8) & 0xF),                             \
                            fa_rocmfp4_code_value((vui >> 16) & 0xF),                             \
                            fa_rocmfp4_code_value((vui >> 24) & 0xF));                            \
}

#define FA_DEQUANT4_ROCMFP4_FAST(BUF) {                                                           \
    const uint qbase = iqs & 0xFu;                                                                \
    uint vui = pack32(u8vec4(BUF.data[a_offset + ib].qs[qbase + 0u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 1u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 2u],                              \
                             BUF.data[a_offset + ib].qs[qbase + 3u]));                            \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui >>= shift;                                                                                \
    const FLOAT_TYPE d = fa_rocmfp4_ue4m3_to_fp_half(BUF.data[a_offset + ib].e);                  \
    return d * FLOAT_TYPEV4(fa_rocmfp4_code_value( vui        & 0xF),                             \
                            fa_rocmfp4_code_value((vui >>  8) & 0xF),                             \
                            fa_rocmfp4_code_value((vui >> 16) & 0xF),                             \
                            fa_rocmfp4_code_value((vui >> 24) & 0xF));                            \
}

#define FA_DEQUANT4_ROCMFPX_FP3(BUF) {                                                            \
    const i8vec4 v = unpack8(fa_rocmfpx_fp3_pack4_qs(BUF.data[a_offset + ib].qs, iqs));           \
    const FLOAT_TYPE d0 = fa_ue4m3_to_fp(BUF.data[a_offset + ib].e[0]);                           \
    const FLOAT_TYPE d1 = fa_ue4m3_to_fp(BUF.data[a_offset + ib].e[1]);                           \
    const uint idx = iqs;                                                                           \
    return FLOAT_TYPEV4(FLOAT_TYPE(v.x) * ((idx + 0u) >= 16u ? d1 : d0),                         \
                        FLOAT_TYPE(v.y) * ((idx + 1u) >= 16u ? d1 : d0),                         \
                        FLOAT_TYPE(v.z) * ((idx + 2u) >= 16u ? d1 : d0),                         \
                        FLOAT_TYPE(v.w) * ((idx + 3u) >= 16u ? d1 : d0));                        \
}

#define FA_DEQUANT4_ROCMFPX_FP6(BUF) {                                                            \
    const i8vec4 v = unpack8(fa_rocmfpx_fp6_pack4_qs(BUF.data[a_offset + ib].qs, iqs));           \
    const FLOAT_TYPE d0 = fa_ue4m3_to_fp(BUF.data[a_offset + ib].e[0]);                           \
    const FLOAT_TYPE d1 = fa_ue4m3_to_fp(BUF.data[a_offset + ib].e[1]);                           \
    const uint idx = iqs;                                                                           \
    return FLOAT_TYPEV4(FLOAT_TYPE(v.x) * ((idx + 0u) >= 16u ? d1 : d0),                         \
                        FLOAT_TYPE(v.y) * ((idx + 1u) >= 16u ? d1 : d0),                         \
                        FLOAT_TYPE(v.z) * ((idx + 2u) >= 16u ? d1 : d0),                         \
                        FLOAT_TYPE(v.w) * ((idx + 3u) >= 16u ? d1 : d0));                        \
}

#define FA_DEQUANT4_ROCMFPX_FP8(BUF) {                                                            \
    const uint idx = iqs;                                                                           \
    const FLOAT_TYPE d = fa_ue4m3_to_fp(BUF.data[a_offset + ib].e);                               \
    return d * FLOAT_TYPEV4(FLOAT_TYPE(int(BUF.data[a_offset + ib].qs[idx + 0u])),                \
                            FLOAT_TYPE(int(BUF.data[a_offset + ib].qs[idx + 1u])),                \
                            FLOAT_TYPE(int(BUF.data[a_offset + ib].qs[idx + 2u])),                \
                            FLOAT_TYPE(int(BUF.data[a_offset + ib].qs[idx + 3u])));               \
}

#define FA_DEQUANT4_TURBO3_0(BUF) {                                                               \
    const uint block = a_offset + ib;                                                             \
    const uint bit_off = iqs * 3u;                                                                \
    const uint byte_idx = bit_off >> 3u;                                                          \
    const uint qbits = (uint(BUF.data[block].qs[byte_idx])                                        \
                     | (uint(BUF.data[block].qs[byte_idx + 1u]) << 8)) >> (bit_off & 7u);          \
    const uint qpack = ( qbits         & 0x00000007u)                                              \
                     | ((qbits <<  5u) & 0x00000700u)                                              \
                     | ((qbits << 10u) & 0x00070000u)                                              \
                     | ((qbits << 15u) & 0x07000000u);                                             \
    const u8vec4 q = unpack8(qpack);                                                              \
    return FLOAT_TYPE(BUF.data[block].d) * fa_turbo3_values(q);                                   \
}

#define FA_DEQUANT4_TURBO4_0(BUF) {                                                               \
    const uint block = a_offset + ib;                                                             \
    const uint byte_idx = iqs >> 1u;                                                              \
    const uint qbits = uint(BUF.data[block].qs[byte_idx])                                         \
                     | (uint(BUF.data[block].qs[byte_idx + 1u]) << 8);                             \
    const uint qpack = ( qbits         & 0x000Fu)                                                  \
                     | ((qbits <<  4u) & 0x0F00u)                                                  \
                     | ((qbits <<  8u) & 0x0F0000u)                                                \
                     | ((qbits << 12u) & 0x0F000000u);                                             \
    const u8vec4 q = unpack8(qpack);                                                              \
    return FLOAT_TYPE(BUF.data[block].d) * fa_turbo4_values(q);                                   \
}

#define FA_DEQUANT4_IQ4_NL(BUF) {                                                                 \
    const uint shift = (iqs & 0x10) >> 2;                                                         \
    const uint qs_i = (iqs & 0xC) >> 1;                                                           \
    const uint qsw = uint(BUF.data[a_offset + ib].qs[qs_i])                                       \
                   | (uint(BUF.data[a_offset + ib].qs[qs_i + 1u]) << 16);                         \
    const FLOAT_TYPE d = FLOAT_TYPE(BUF.data[a_offset + ib].d);                                   \
    const u8vec4 q = unpack8((qsw >> shift) & 0x0F0F0F0Fu);                                       \
    return d * FLOAT_TYPEV4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],                               \
                            kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]);                              \
}

#define FA_DEQUANT4_Q1_0(BUF) {                                                                   \
    const uint bits = uint(BUF.data[a_offset + ib].qs[iqs / 8u]) >> (iqs % 8u);                   \
    const FLOAT_TYPE d = FLOAT_TYPE(BUF.data[a_offset + ib].d);                                   \
    return d * FLOAT_TYPEV4((bits & 1u) != 0u ? 1.0f : -1.0f,                                     \
                            (bits & 2u) != 0u ? 1.0f : -1.0f,                                     \
                            (bits & 4u) != 0u ? 1.0f : -1.0f,                                     \
                            (bits & 8u) != 0u ? 1.0f : -1.0f);                                    \
}

#define FA_DEQUANT4_BF16(BUF) \
    return FLOAT_TYPEV4(bf16_to_fp32(uvec4(BUF.data[(a_offset + ib) / 4])));

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        switch (FaTypeK) {
            case FA_TYPE_F32:  FA_DEQUANT4_F32 (k_packed_f32)
            case FA_TYPE_Q4_0: FA_DEQUANT4_Q4_0(k_packed_q4_0)
            case FA_TYPE_Q4_1: FA_DEQUANT4_Q4_1(k_packed_q4_1)
            case FA_TYPE_Q5_0: FA_DEQUANT4_Q5_0(k_packed_q5_0)
            case FA_TYPE_Q5_1: FA_DEQUANT4_Q5_1(k_packed_q5_1)
            case FA_TYPE_Q8_0: FA_DEQUANT4_Q8_0(k_packed_q8_0)
            case FA_TYPE_Q4_0_ROCMFP4:      FA_DEQUANT4_ROCMFP4(k_packed_rocmfp4)
            case FA_TYPE_Q4_0_ROCMFP4_FAST: FA_DEQUANT4_ROCMFP4_FAST(k_packed_rocmfp4_fast)
            case FA_TYPE_Q3_0_ROCMFPX:      FA_DEQUANT4_ROCMFPX_FP3(k_packed_rocmfpx_fp3)
            case FA_TYPE_Q6_0_ROCMFPX:      FA_DEQUANT4_ROCMFPX_FP6(k_packed_rocmfpx_fp6)
            case FA_TYPE_Q8_0_ROCMFPX:      FA_DEQUANT4_ROCMFPX_FP8(k_packed_rocmfpx_fp8)
            case FA_TYPE_TURBO3_0: FA_DEQUANT4_TURBO3_0(k_packed_turbo3_0)
            case FA_TYPE_TURBO4_0: FA_DEQUANT4_TURBO4_0(k_packed_turbo4_0)
            case FA_TYPE_IQ4_NL: FA_DEQUANT4_IQ4_NL(k_packed_iq4_nl)
            case FA_TYPE_BF16: FA_DEQUANT4_BF16(k_packed_bf16)
            case FA_TYPE_Q1_0: FA_DEQUANT4_Q1_0(k_packed_q1_0)
        }
    } else {
        switch (FaTypeV) {
            case FA_TYPE_F32:  FA_DEQUANT4_F32 (v_packed_f32)
            case FA_TYPE_Q4_0: FA_DEQUANT4_Q4_0(v_packed_q4_0)
            case FA_TYPE_Q4_1: FA_DEQUANT4_Q4_1(v_packed_q4_1)
            case FA_TYPE_Q5_0: FA_DEQUANT4_Q5_0(v_packed_q5_0)
            case FA_TYPE_Q5_1: FA_DEQUANT4_Q5_1(v_packed_q5_1)
            case FA_TYPE_Q8_0: FA_DEQUANT4_Q8_0(v_packed_q8_0)
            case FA_TYPE_Q4_0_ROCMFP4:      FA_DEQUANT4_ROCMFP4(v_packed_rocmfp4)
            case FA_TYPE_Q4_0_ROCMFP4_FAST: FA_DEQUANT4_ROCMFP4_FAST(v_packed_rocmfp4_fast)
            case FA_TYPE_Q3_0_ROCMFPX:      FA_DEQUANT4_ROCMFPX_FP3(v_packed_rocmfpx_fp3)
            case FA_TYPE_Q6_0_ROCMFPX:      FA_DEQUANT4_ROCMFPX_FP6(v_packed_rocmfpx_fp6)
            case FA_TYPE_Q8_0_ROCMFPX:      FA_DEQUANT4_ROCMFPX_FP8(v_packed_rocmfpx_fp8)
            case FA_TYPE_TURBO3_0: FA_DEQUANT4_TURBO3_0(v_packed_turbo3_0)
            case FA_TYPE_TURBO4_0: FA_DEQUANT4_TURBO4_0(v_packed_turbo4_0)
            case FA_TYPE_IQ4_NL: FA_DEQUANT4_IQ4_NL(v_packed_iq4_nl)
            case FA_TYPE_BF16: FA_DEQUANT4_BF16(v_packed_bf16)
            case FA_TYPE_Q1_0: FA_DEQUANT4_Q1_0(v_packed_q1_0)
        }
    }
    return FLOAT_TYPEV4(0);
}
