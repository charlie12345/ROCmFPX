// MMQ V-side helpers, asymmetric form. Mirrors flash_attn_mmq_funcs.glsl but
// dispatches on FaTypeV and reads from the V bindings in flash_attn_dequant.glsl.

int32_t get_v_qs(uint ib, uint iqs, uint a_offset) {
    switch (FaTypeV) {
        case FA_TYPE_Q4_0_ROCMFP4: {
            const uint qbase = iqs & 0xFu;
            uint vui = pack32(u8vec4(v_packed_rocmfp4.data[a_offset + ib].qs[qbase + 0u],
                                     v_packed_rocmfp4.data[a_offset + ib].qs[qbase + 1u],
                                     v_packed_rocmfp4.data[a_offset + ib].qs[qbase + 2u],
                                     v_packed_rocmfp4.data[a_offset + ib].qs[qbase + 3u]));
            uint shift = (iqs & 0x10) >> 2;
            return fa_rocmfp4_pack4_i8(vui >> shift);
        }
        case FA_TYPE_Q4_0_ROCMFP4_FAST: {
            const uint qbase = iqs & 0xFu;
            uint vui = pack32(u8vec4(v_packed_rocmfp4_fast.data[a_offset + ib].qs[qbase + 0u],
                                     v_packed_rocmfp4_fast.data[a_offset + ib].qs[qbase + 1u],
                                     v_packed_rocmfp4_fast.data[a_offset + ib].qs[qbase + 2u],
                                     v_packed_rocmfp4_fast.data[a_offset + ib].qs[qbase + 3u]));
            uint shift = (iqs & 0x10) >> 2;
            return fa_rocmfp4_pack4_i8(vui >> shift);
        }
        case FA_TYPE_Q3_0_ROCMFPX:
            return fa_rocmfpx_fp3_pack4_qs(v_packed_rocmfpx_fp3.data[a_offset + ib].qs, iqs);
        case FA_TYPE_Q6_0_ROCMFPX:
            return fa_rocmfpx_fp6_pack4_qs(v_packed_rocmfpx_fp6.data[a_offset + ib].qs, iqs);
        case FA_TYPE_Q8_0_ROCMFPX:
            return fa_rocmfpx_fp8_pack4_qs(v_packed_rocmfpx_fp8.data[a_offset + ib].qs, iqs);
        default: return 0;
    }
}

FLOAT_TYPEV2 get_v_scale(uint ib, uint a_offset) {
    switch (FaTypeV) {
        case FA_TYPE_Q4_0_ROCMFP4:
            return FLOAT_TYPEV2(fa_rocmfp4_ue4m3_to_fp_half(v_packed_rocmfp4.data[a_offset + ib].e[0]),
                                fa_rocmfp4_ue4m3_to_fp_half(v_packed_rocmfp4.data[a_offset + ib].e[1]));
        case FA_TYPE_Q4_0_ROCMFP4_FAST:
            return FLOAT_TYPEV2(fa_rocmfp4_ue4m3_to_fp_half(v_packed_rocmfp4_fast.data[a_offset + ib].e), 0.0);
        case FA_TYPE_Q3_0_ROCMFPX:
            return FLOAT_TYPEV2(fa_ue4m3_to_fp(v_packed_rocmfpx_fp3.data[a_offset + ib].e[0]),
                                fa_ue4m3_to_fp(v_packed_rocmfpx_fp3.data[a_offset + ib].e[1]));
        case FA_TYPE_Q6_0_ROCMFPX:
            return FLOAT_TYPEV2(fa_ue4m3_to_fp(v_packed_rocmfpx_fp6.data[a_offset + ib].e[0]),
                                fa_ue4m3_to_fp(v_packed_rocmfpx_fp6.data[a_offset + ib].e[1]));
        case FA_TYPE_Q8_0_ROCMFPX:
            return FLOAT_TYPEV2(fa_ue4m3_to_fp(v_packed_rocmfpx_fp8.data[a_offset + ib].e), 0.0);
        default: return FLOAT_TYPEV2(0);
    }
}

struct fa_v_qs_block8 {
    int32_t qs[8];
};

fa_v_qs_block8 get_v_qs_block8(uint ib, uint a_offset) {
    fa_v_qs_block8 r;
    if (FaTypeV == FA_TYPE_Q3_0_ROCMFPX) {
        [[unroll]] for (uint32_t d = 0; d < 8; d++) {
            r.qs[d] = fa_rocmfpx_fp3_pack4_qs(v_packed_rocmfpx_fp3.data[a_offset + ib].qs, d * 4u);
        }
        return r;
    }
    if (FaTypeV == FA_TYPE_Q6_0_ROCMFPX) {
        [[unroll]] for (uint32_t d = 0; d < 8; d++) {
            r.qs[d] = fa_rocmfpx_fp6_pack4_qs(v_packed_rocmfpx_fp6.data[a_offset + ib].qs, d * 4u);
        }
        return r;
    }
    if (FaTypeV == FA_TYPE_Q8_0_ROCMFPX) {
        [[unroll]] for (uint32_t d = 0; d < 8; d++) {
            r.qs[d] = fa_rocmfpx_fp8_pack4_qs(v_packed_rocmfpx_fp8.data[a_offset + ib].qs, d * 4u);
        }
        return r;
    }
    if (FaTypeV == FA_TYPE_Q4_0_ROCMFP4 || FaTypeV == FA_TYPE_Q4_0_ROCMFP4_FAST) {
        [[unroll]] for (uint32_t d = 0; d < 4; d++) {
            const uint qbase = d * 4u;
            uint vui;
            if (FaTypeV == FA_TYPE_Q4_0_ROCMFP4) {
                vui = pack32(u8vec4(v_packed_rocmfp4.data[a_offset + ib].qs[qbase + 0u],
                                    v_packed_rocmfp4.data[a_offset + ib].qs[qbase + 1u],
                                    v_packed_rocmfp4.data[a_offset + ib].qs[qbase + 2u],
                                    v_packed_rocmfp4.data[a_offset + ib].qs[qbase + 3u]));
            } else {
                vui = pack32(u8vec4(v_packed_rocmfp4_fast.data[a_offset + ib].qs[qbase + 0u],
                                    v_packed_rocmfp4_fast.data[a_offset + ib].qs[qbase + 1u],
                                    v_packed_rocmfp4_fast.data[a_offset + ib].qs[qbase + 2u],
                                    v_packed_rocmfp4_fast.data[a_offset + ib].qs[qbase + 3u]));
            }
            r.qs[d    ] = fa_rocmfp4_pack4_i8( vui       & 0x0F0F0F0Fu);
            r.qs[d + 4] = fa_rocmfp4_pack4_i8((vui >> 4) & 0x0F0F0F0Fu);
        }
        return r;
    }
    return r;
}

FLOAT_TYPEV4 fa_v_mmq_vec4_from_pack(const int32_t v_pack, const FLOAT_TYPEV2 dm, const uint v_sub) {
    const i8vec4 v = unpack8(v_pack);
    const uint base = v_sub * 4u;
    switch (FaTypeV) {
        case FA_TYPE_Q3_0_ROCMFPX:
        case FA_TYPE_Q6_0_ROCMFPX:
        case FA_TYPE_Q4_0_ROCMFP4:
            return FLOAT_TYPEV4(
                FLOAT_TYPE(v.x) * ((base + 0u) >= 16u ? dm.y : dm.x),
                FLOAT_TYPE(v.y) * ((base + 1u) >= 16u ? dm.y : dm.x),
                FLOAT_TYPE(v.z) * ((base + 2u) >= 16u ? dm.y : dm.x),
                FLOAT_TYPE(v.w) * ((base + 3u) >= 16u ? dm.y : dm.x));
        case FA_TYPE_Q4_0_ROCMFP4_FAST:
        case FA_TYPE_Q8_0_ROCMFPX:
            return FLOAT_TYPE(dm.x) * FLOAT_TYPEV4(FLOAT_TYPE(v.x), FLOAT_TYPE(v.y), FLOAT_TYPE(v.z), FLOAT_TYPE(v.w));
        default:
            return FLOAT_TYPEV4(FLOAT_TYPE(0.0), FLOAT_TYPE(0.0), FLOAT_TYPE(0.0), FLOAT_TYPE(0.0));
    }
}

// Stage one 32-element V block into kvsh with a single qs load (8 vec4 outputs).
void fa_v_mmq_stage_block_kvsh(const uint c, const uint block, const uint a_offset, const uint kv_base_col) {
    const uint global_ib = kv_base_col * v_stride + block;
    const fa_v_qs_block8 blk = get_v_qs_block8(global_ib, a_offset);
    const FLOAT_TYPEV2 dm = get_v_scale(global_ib, a_offset);
    [[unroll]] for (uint sub = 0u; sub < 8u; ++sub) {
        kvsh[c * kvsh_stride + block * 8u + sub] = fa_v_mmq_vec4_from_pack(blk.qs[sub], dm, sub);
    }
}
