#pragma once
#include "simd_types.h"
#include <common/defines.h>
#include <common/tupel.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace opp::cuda::simd
{
template <typename T> struct Neg
{
    static constexpr bool has_simd = IsSignedSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vnegss4(src1[0]);
        dst[1] = __vnegss4(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        uint *srcdst = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vnegss4(srcdst[0]);
        srcdst[1] = __vnegss4(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vnegss2(src1[0]);
        dst[1] = __vnegss2(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, short1_4>
    {
        uint *srcdst = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vnegss2(srcdst[0]);
        srcdst[1] = __vnegss2(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = __hneg2(src1[0]);
        dst[1] = __hneg2(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, bfloat1_4>
    {
        nv_bfloat162 *srcdst = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __hneg2(srcdst[0]);
        srcdst[1] = __hneg2(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = __hneg2(src1[0]);
        dst[1] = __hneg2(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, hfloat1_4>
    {
        half2 *srcdst = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __hneg2(srcdst[0]);
        srcdst[1] = __hneg2(srcdst[1]);
    }
};

template <typename T> struct Abs
{
    static constexpr bool has_simd = IsSignedSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vabsss4(src1[0]);
        dst[1] = __vabsss4(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        uint *srcdst = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsss4(srcdst[0]);
        srcdst[1] = __vabsss4(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, short1_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vabsss2(src1[0]);
        dst[1] = __vabsss2(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, short1_4>
    {
        uint *srcdst = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsss2(srcdst[0]);
        srcdst[1] = __vabsss2(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = __habs2(src1[0]);
        dst[1] = __habs2(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, bfloat1_4>
    {
        nv_bfloat162 *srcdst = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = __habs2(srcdst[0]);
        srcdst[1] = __habs2(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = __habs2(src1[0]);
        dst[1] = __habs2(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, hfloat1_4>
    {
        half2 *srcdst = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = __habs2(srcdst[0]);
        srcdst[1] = __habs2(srcdst[1]);
    }
};

template <typename T> struct Exp
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = h2exp(src1[0]);
        dst[1] = h2exp(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, bfloat1_4>
    {
        nv_bfloat162 *srcdst = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = h2exp(srcdst[0]);
        srcdst[1] = h2exp(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = h2exp(src1[0]);
        dst[1] = h2exp(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, hfloat1_4>
    {
        half2 *srcdst = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = h2exp(srcdst[0]);
        srcdst[1] = h2exp(srcdst[1]);
    }
};

template <typename T> struct Ln
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = h2log(src1[0]);
        dst[1] = h2log(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, bfloat1_4>
    {
        nv_bfloat162 *srcdst = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = h2log(srcdst[0]);
        srcdst[1] = h2log(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = h2log(src1[0]);
        dst[1] = h2log(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, hfloat1_4>
    {
        half2 *srcdst = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = h2log(srcdst[0]);
        srcdst[1] = h2log(srcdst[1]);
    }
};

template <typename T> struct Sqrt
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = h2sqrt(src1[0]);
        dst[1] = h2sqrt(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, bfloat1_4>
    {
        nv_bfloat162 *srcdst = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = h2sqrt(srcdst[0]);
        srcdst[1] = h2sqrt(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = h2sqrt(src1[0]);
        dst[1] = h2sqrt(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, hfloat1_4>
    {
        half2 *srcdst = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = h2sqrt(srcdst[0]);
        srcdst[1] = h2sqrt(srcdst[1]);
    }
};

template <typename T> struct Floor
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = h2floor(aSrc1[0]);
        dst[1] = h2floor(aSrc1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, bfloat1_4>
    {
        nv_bfloat162 *srcdst = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = h2floor(srcdst[0]);
        srcdst[1] = h2floor(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = h2floor(src1[0]);
        dst[1] = h2floor(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, hfloat1_4>
    {
        half2 *srcdst = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = h2floor(srcdst[0]);
        srcdst[1] = h2floor(srcdst[1]);
    }
};

template <typename T> struct Ceil
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = h2ceil(src1[0]);
        dst[1] = h2ceil(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, bfloat1_4>
    {
        nv_bfloat162 *srcdst = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = h2ceil(srcdst[0]);
        srcdst[1] = h2ceil(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = h2ceil(src1[0]);
        dst[1] = h2ceil(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, hfloat1_4>
    {
        half2 *srcdst = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = h2ceil(srcdst[0]);
        srcdst[1] = h2ceil(srcdst[1]);
    }
};

template <typename T> struct RoundNearest
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = h2rint(src1[0]);
        dst[1] = h2rint(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, bfloat1_4>
    {
        nv_bfloat162 *srcdst = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = h2rint(srcdst[0]);
        srcdst[1] = h2rint(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = h2rint(src1[0]);
        dst[1] = h2rint(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, hfloat1_4>
    {
        half2 *srcdst = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = h2rint(srcdst[0]);
        srcdst[1] = h2rint(srcdst[1]);
    }
};

template <typename T> struct RoundZero
{
    static constexpr bool has_simd = IsNonNativeSimdType<T>;

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, bfloat1_4>
    {
        const nv_bfloat162 *src1 = reinterpret_cast<const nv_bfloat162 *>(&aSrc1);
        nv_bfloat162 *dst        = reinterpret_cast<nv_bfloat162 *>(&aDst);

        dst[0] = h2trunc(src1[0]);
        dst[1] = h2trunc(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, bfloat1_4>
    {
        nv_bfloat162 *srcdst = reinterpret_cast<nv_bfloat162 *>(&aSrcDst);

        srcdst[0] = h2trunc(srcdst[0]);
        srcdst[1] = h2trunc(srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aDst)
        requires std::same_as<T, hfloat1_4>
    {
        const half2 *src1 = reinterpret_cast<const half2 *>(&aSrc1);
        half2 *dst        = reinterpret_cast<half2 *>(&aDst);

        dst[0] = h2trunc(src1[0]);
        dst[1] = h2trunc(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, hfloat1_4>
    {
        half2 *srcdst = reinterpret_cast<half2 *>(&aSrcDst);

        srcdst[0] = h2trunc(srcdst[0]);
        srcdst[1] = h2trunc(srcdst[1]);
    }
};

template <typename TTo> struct Convert
{
    static constexpr bool has_simd = IsNonNativeSimdType<TTo>;

    DEVICE_ONLY_CODE void operator()(const Tupel<Vector1<float>, 4> &aSrc1, TTo &aDst)
        requires std::same_as<TTo, bfloat1_4>
    {
        const float2 *src1 = reinterpret_cast<const float2 *>(&aSrc1);
        nv_bfloat162 *dst  = reinterpret_cast<nv_bfloat162 *>(&aDst);
        dst[0]             = __float22bfloat162_rn(src1[0]);
        dst[1]             = __float22bfloat162_rn(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(const Tupel<Vector1<float>, 4> &aSrc1, TTo &aDst)
        requires std::same_as<TTo, hfloat1_4>
    {
        const float2 *src1 = reinterpret_cast<const float2 *>(&aSrc1);
        half2 *dst         = reinterpret_cast<half2 *>(&aDst);
        dst[0]             = __float22half2_rn(src1[0]);
        dst[1]             = __float22half2_rn(src1[1]);
    }
};
} // namespace opp::cuda::simd
