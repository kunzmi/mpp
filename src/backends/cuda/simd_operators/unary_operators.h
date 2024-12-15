#pragma once
#include <common/defines.h>
#include <common/tupel.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp::cuda::simd
{
using sbyte1_8 = Tupel<Vector1<sbyte>, 8>;
using sbyte2_4 = Tupel<Vector2<sbyte>, 4>;
using short1_4 = Tupel<Vector1<short>, 4>;
using short2_2 = Tupel<Vector2<short>, 2>;

template <VectorType T> struct Neg
{
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
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vnegss2(src1[0]);
        dst[1] = __vnegss2(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        uint *srcdst = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vnegss2(srcdst[0]);
        srcdst[1] = __vnegss2(srcdst[1]);
    }
};

template <VectorType T> struct Abs
{
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
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        const uint *src1 = reinterpret_cast<const uint *>(&aSrc1);
        uint *dst        = reinterpret_cast<uint *>(&aDst);

        dst[0] = __vabsss2(src1[0]);
        dst[1] = __vabsss2(src1[1]);
    }
    DEVICE_ONLY_CODE void operator()(T &aSrcDst)
        requires std::same_as<T, short1_4> || std::same_as<T, short2_2>
    {
        uint *srcdst = reinterpret_cast<uint *>(&aSrcDst);

        srcdst[0] = __vabsss2(srcdst[0]);
        srcdst[1] = __vabsss2(srcdst[1]);
    }
};
} // namespace opp::cuda::simd
