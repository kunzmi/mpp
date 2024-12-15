#pragma once
#include <common/defines.h>
#include <common/tupel.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp::cuda::simd
{
using byte1_8   = Tupel<Vector1<byte>, 8>;
using sbyte1_8  = Tupel<Vector1<sbyte>, 8>;
using byte2_4   = Tupel<Vector2<byte>, 4>;
using sbyte2_4  = Tupel<Vector2<sbyte>, 4>;
using ushort1_4 = Tupel<Vector1<ushort>, 4>;
using short1_4  = Tupel<Vector1<short>, 4>;
using ushort2_2 = Tupel<Vector2<ushort>, 2>;
using short2_2  = Tupel<Vector2<short>, 2>;

template <typename T> struct AbsDiff
{
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vabsdiffu4(src1[0], src2[0]);
        dst[1] = __vabsdiffu4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsdiffu4(src1[0], srcdst[0]);
        srcdst[1] = __vabsdiffu4(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vabsdiffs4(src1[0], src2[0]);
        dst[1] = __vabsdiffs4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsdiffs4(src1[0], srcdst[0]);
        srcdst[1] = __vabsdiffs4(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, ushort1_4> || std::same_as<T, ushort2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vabsdiffu2(src1[0], src2[0]);
        dst[1] = __vabsdiffu2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, ushort1_4> || std::same_as<T, ushort2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsdiffu2(src1[0], srcdst[0]);
        srcdst[1] = __vabsdiffu2(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(aDst);

        dst[0] = __vabsdiffs2(src1[0], src2[0]);
        dst[1] = __vabsdiffs2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsdiffs2(src1[0], srcdst[0]);
        srcdst[1] = __vabsdiffs2(src1[1], srcdst[1]);
    }
};

template <typename T> struct Add
{
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vaddus4(src1[0], src2[0]);
        dst[1] = __vaddus4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vaddus4(src1[0], srcdst[0]);
        srcdst[1] = __vaddus4(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vaddss4(src1[0], src2[0]);
        dst[1] = __vaddss4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vaddss4(src1[0], srcdst[0]);
        srcdst[1] = __vaddss4(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, ushort1_4> || std::same_as<T, ushort2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vaddus2(src1[0], src2[0]);
        dst[1] = __vaddus2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, ushort1_4> || std::same_as<T, ushort2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vaddus2(src1[0], srcdst[0]);
        srcdst[1] = __vaddus2(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vaddss2(src1[0], src2[0]);
        dst[1] = __vaddss2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vaddss2(src1[0], srcdst[0]);
        srcdst[1] = __vaddss2(src1[1], srcdst[1]);
    }
};

template <typename T> struct Sub
{
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vsubus4(src1[0], src2[0]);
        dst[1] = __vsubus4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubus4(src1[0], srcdst[0]);
        srcdst[1] = __vsubus4(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vsubss4(src1[0], src2[0]);
        dst[1] = __vsubss4(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubss4(src1[0], srcdst[0]);
        srcdst[1] = __vsubss4(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, ushort1_4> || std::same_as<T, ushort2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vsubus2(src1[0], src2[0]);
        dst[1] = __vsubus2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, ushort1_4> || std::same_as<T, ushort2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubus2(src1[0], srcdst[0]);
        srcdst[1] = __vsubus2(src1[1], srcdst[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        const uint *src2 = reinterpret_cast<const uint *>(&aSrc2);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vsubss2(src1[0], src2[0]);
        dst[1] = __vsubss2(src1[1], src2[1]);
    }
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubss2(src1[0], srcdst[0]);
        srcdst[1] = __vsubss2(src1[1], srcdst[1]);
    }
};

//
///// <summary>
///// Inverted argument order for inplace sub: aSrcDst = aSrc1 - aSrcDst
///// </summary>
template <typename T> struct SubInv
{
    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, byte1_8> || std::same_as<T, byte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubus4(srcdst[0], src1[0]);
        srcdst[1] = __vsubus4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubss4(srcdst[0], src1[0]);
        srcdst[1] = __vsubss4(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, ushort1_4> || std::same_as<T, ushort2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubus2(srcdst[0], src1[0]);
        srcdst[1] = __vsubus2(srcdst[1], src1[1]);
    }

    DEVICE_ONLY_CODE void operator()(const T &aSrc1, T &aSrcDst)
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *srcdst     = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vsubss2(srcdst[0], src1[0]);
        srcdst[1] = __vsubss2(srcdst[1], src1[1]);
    }
};

} // namespace opp::cuda::simd
