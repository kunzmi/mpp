#pragma once
#include "simd_types.h"
#include <common/defines.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace mpp::cuda::simd
{

template <typename T> struct AbsDiff
{
    static constexpr bool has_simd = IsNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vabsdiffu4(src1[0], src2[0]);
        dst[1] = __vabsdiffu4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsdiffu4(srcdst[0], src1[0]);
        srcdst[1] = __vabsdiffu4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vabsdiffs4(src1[0], src2[0]);
        dst[1] = __vabsdiffs4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsdiffs4(srcdst[0], src1[0]);
        srcdst[1] = __vabsdiffs4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vabsdiffu2(src1[0], src2[0]);
        dst[1] = __vabsdiffu2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsdiffu2(srcdst[0], src1[0]);
        srcdst[1] = __vabsdiffu2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(aDst);

        dst[0] = __vabsdiffs2(src1[0], src2[0]);
        dst[1] = __vabsdiffs2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsdiffs2(srcdst[0], src1[0]);
        srcdst[1] = __vabsdiffs2(srcdst[1], src1[1]);
    }
};

template <typename T> struct Add
{
    static constexpr bool has_simd = IsSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vaddus4(src1[0], src2[0]);
        dst[1] = __vaddus4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vaddus4(srcdst[0], src1[0]);
        srcdst[1] = __vaddus4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vaddss4(src1[0], src2[0]);
        dst[1] = __vaddss4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vaddss4(srcdst[0], src1[0]);
        srcdst[1] = __vaddss4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vaddus2(src1[0], src2[0]);
        dst[1] = __vaddus2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vaddus2(srcdst[0], src1[0]);
        srcdst[1] = __vaddus2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vaddss2(src1[0], src2[0]);
        dst[1] = __vaddss2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vaddss2(srcdst[0], src1[0]);
        srcdst[1] = __vaddss2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        const nv_bfloat162 *src2 = reinterpret_cast<const nv_bfloat162 *>(&aSrc2);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = __hadd2(src1[0], src2[0]);
        dst[1] = __hadd2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *srcdst     = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __hadd2(srcdst[0], src1[0]);
        srcdst[1] = __hadd2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        const half2 *src2 = reinterpret_cast<const half2 *>(&aSrc2);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = __hadd2(src1[0], src2[0]);
        dst[1] = __hadd2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *srcdst     = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __hadd2(srcdst[0], src1[0]);
        srcdst[1] = __hadd2(srcdst[1], src1[1]);
    }
};

template <typename T> struct Sub
{
    static constexpr bool has_simd = IsSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vsubus4(src1[0], src2[0]);
        dst[1] = __vsubus4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubus4(srcdst[0], src1[0]);
        srcdst[1] = __vsubus4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vsubss4(src1[0], src2[0]);
        dst[1] = __vsubss4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubss4(srcdst[0], src1[0]);
        srcdst[1] = __vsubss4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vsubus2(src1[0], src2[0]);
        dst[1] = __vsubus2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubus2(srcdst[0], src1[0]);
        srcdst[1] = __vsubus2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vsubss2(src1[0], src2[0]);
        dst[1] = __vsubss2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubss2(srcdst[0], src1[0]);
        srcdst[1] = __vsubss2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        const nv_bfloat162 *src2 = reinterpret_cast<const nv_bfloat162 *>(&aSrc2);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = __hsub2(src1[0], src2[0]);
        dst[1] = __hsub2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *srcdst     = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __hsub2(srcdst[0], src1[0]);
        srcdst[1] = __hsub2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        const half2 *src2 = reinterpret_cast<const half2 *>(&aSrc2);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = __hsub2(src1[0], src2[0]);
        dst[1] = __hsub2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *srcdst     = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __hsub2(srcdst[0], src1[0]);
        srcdst[1] = __hsub2(srcdst[1], src1[1]);
    }
};

///// <summary>
///// Inverted argument order for inplace sub: aSrcDst = aSrc1 - aSrcDst
///// </summary>
template <typename T> struct SubInv
{
    static constexpr bool has_simd = IsSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubus4(src1[0], srcdst[0]);
        srcdst[1] = __vsubus4(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubss4(src1[0], srcdst[0]);
        srcdst[1] = __vsubss4(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubus2(src1[0], srcdst[0]);
        srcdst[1] = __vsubus2(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubss2(src1[0], srcdst[0]);
        srcdst[1] = __vsubss2(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *srcdst     = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __hsub2(src1[0], srcdst[0]);
        srcdst[1] = __hsub2(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *srcdst     = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __hsub2(src1[0], srcdst[0]);
        srcdst[1] = __hsub2(src1[1], srcdst[1]);
    }
};

template <typename T> struct Mul
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        const nv_bfloat162 *src2 = reinterpret_cast<const nv_bfloat162 *>(&aSrc2);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = __hmul2(src1[0], src2[0]);
        dst[1] = __hmul2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *srcdst     = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __hmul2(srcdst[0], src1[0]);
        srcdst[1] = __hmul2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        const half2 *src2 = reinterpret_cast<const half2 *>(&aSrc2);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = __hmul2(src1[0], src2[0]);
        dst[1] = __hmul2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *srcdst     = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __hmul2(srcdst[0], src1[0]);
        srcdst[1] = __hmul2(srcdst[1], src1[1]);
    }
};

template <typename T> struct Div
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        const nv_bfloat162 *src2 = reinterpret_cast<const nv_bfloat162 *>(&aSrc2);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = __h2div(src1[0], src2[0]);
        dst[1] = __h2div(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *srcdst     = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __h2div(srcdst[0], src1[0]);
        srcdst[1] = __h2div(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        const half2 *src2 = reinterpret_cast<const half2 *>(&aSrc2);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = __h2div(src1[0], src2[0]);
        dst[1] = __h2div(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *srcdst     = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __h2div(srcdst[0], src1[0]);
        srcdst[1] = __h2div(srcdst[1], src1[1]);
    }
};

///// <summary>
///// Inverted argument order for inplace div: aSrcDst = aSrc1 / aSrcDst
///// </summary>
template <typename T> struct DivInv
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *srcdst     = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __h2div(src1[0], srcdst[0]);
        srcdst[1] = __h2div(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *srcdst     = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __h2div(src1[0], srcdst[0]);
        srcdst[1] = __h2div(src1[1], srcdst[1]);
    }
};

template <typename T> struct Min
{
    static constexpr bool has_simd = IsSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vminu4(src1[0], src2[0]);
        dst[1] = __vminu4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vminu4(srcdst[0], src1[0]);
        srcdst[1] = __vminu4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vmins4(src1[0], src2[0]);
        dst[1] = __vmins4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vmins4(srcdst[0], src1[0]);
        srcdst[1] = __vmins4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vminu2(src1[0], src2[0]);
        dst[1] = __vminu2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vminu2(srcdst[0], src1[0]);
        srcdst[1] = __vminu2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vmins2(src1[0], src2[0]);
        dst[1] = __vmins2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vmins2(srcdst[0], src1[0]);
        srcdst[1] = __vmins2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        const nv_bfloat162 *src2 = reinterpret_cast<const nv_bfloat162 *>(&aSrc2);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = __hmin2(src1[0], src2[0]);
        dst[1] = __hmin2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *srcdst     = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __hmin2(srcdst[0], src1[0]);
        srcdst[1] = __hmin2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        const half2 *src2 = reinterpret_cast<const half2 *>(&aSrc2);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = __hmin2(src1[0], src2[0]);
        dst[1] = __hmin2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *srcdst     = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __hmin2(srcdst[0], src1[0]);
        srcdst[1] = __hmin2(srcdst[1], src1[1]);
    }
};

template <typename T> struct Max
{
    static constexpr bool has_simd = IsSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vmaxu4(src1[0], src2[0]);
        dst[1] = __vmaxu4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vmaxu4(srcdst[0], src1[0]);
        srcdst[1] = __vmaxu4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vmaxs4(src1[0], src2[0]);
        dst[1] = __vmaxs4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vmaxs4(srcdst[0], src1[0]);
        srcdst[1] = __vmaxs4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vmaxu2(src1[0], src2[0]);
        dst[1] = __vmaxu2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, ushort1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vmaxu2(srcdst[0], src1[0]);
        srcdst[1] = __vmaxu2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vmaxs2(src1[0], src2[0]);
        dst[1] = __vmaxs2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vmaxs2(srcdst[0], src1[0]);
        srcdst[1] = __vmaxs2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        const nv_bfloat162 *src2 = reinterpret_cast<const nv_bfloat162 *>(&aSrc2);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = __hmax2(src1[0], src2[0]);
        dst[1] = __hmax2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *srcdst     = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __hmax2(srcdst[0], src1[0]);
        srcdst[1] = __hmax2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        const half2 *src2 = reinterpret_cast<const half2 *>(&aSrc2);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = __hmax2(src1[0], src2[0]);
        dst[1] = __hmax2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *srcdst     = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __hmax2(srcdst[0], src1[0]);
        srcdst[1] = __hmax2(srcdst[1], src1[1]);
    }
};
} // namespace mpp::cuda::simd
