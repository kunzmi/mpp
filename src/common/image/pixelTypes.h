#pragma once
#include <common/complex.h>
#include <common/complex_typetraits.h>
#include <common/defines.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <ostream>

namespace opp
{
// forward declaration:
class HalfFp16;
class BFloat16;
} // namespace opp

namespace opp::image
{
#pragma region Type definitions
using Pixel64fC1  = Vector1<double>;
using Pixel64fC2  = Vector2<double>;
using Pixel64fC3  = Vector3<double>;
using Pixel64fC4  = Vector4<double>;
using Pixel64fC4A = Vector4A<double>;

using Pixel64fcC1 = Vector1<c_double>;
using Pixel64fcC2 = Vector2<c_double>;
using Pixel64fcC3 = Vector3<c_double>;
using Pixel64fcC4 = Vector4<c_double>;

using Pixel32fC1  = Vector1<float>;
using Pixel32fC2  = Vector2<float>;
using Pixel32fC3  = Vector3<float>;
using Pixel32fC4  = Vector4<float>;
using Pixel32fC4A = Vector4A<float>;

using Pixel32fcC1 = Vector1<c_float>;
using Pixel32fcC2 = Vector2<c_float>;
using Pixel32fcC3 = Vector3<c_float>;
using Pixel32fcC4 = Vector4<c_float>;

using Pixel32sC1  = Vector1<int>;
using Pixel32sC2  = Vector2<int>;
using Pixel32sC3  = Vector3<int>;
using Pixel32sC4  = Vector4<int>;
using Pixel32sC4A = Vector4A<int>;

using Pixel32scC1 = Vector1<c_int>;
using Pixel32scC2 = Vector2<c_int>;
using Pixel32scC3 = Vector3<c_int>;
using Pixel32scC4 = Vector4<c_int>;

using Pixel32uC1  = Vector1<uint>;
using Pixel32uC2  = Vector2<uint>;
using Pixel32uC3  = Vector3<uint>;
using Pixel32uC4  = Vector4<uint>;
using Pixel32uC4A = Vector4A<uint>;

using Pixel16fC1  = Vector1<HalfFp16>;
using Pixel16fC2  = Vector2<HalfFp16>;
using Pixel16fC3  = Vector3<HalfFp16>;
using Pixel16fC4  = Vector4<HalfFp16>;
using Pixel16fC4A = Vector4A<HalfFp16>;

using Pixel16fcC1 = Vector1<c_HalfFp16>;
using Pixel16fcC2 = Vector2<c_HalfFp16>;
using Pixel16fcC3 = Vector3<c_HalfFp16>;
using Pixel16fcC4 = Vector4<c_HalfFp16>;

using Pixel16bfC1  = Vector1<BFloat16>;
using Pixel16bfC2  = Vector2<BFloat16>;
using Pixel16bfC3  = Vector3<BFloat16>;
using Pixel16bfC4  = Vector4<BFloat16>;
using Pixel16bfC4A = Vector4A<BFloat16>;

using Pixel16bfcC1 = Vector1<c_BFloat16>;
using Pixel16bfcC2 = Vector2<c_BFloat16>;
using Pixel16bfcC3 = Vector3<c_BFloat16>;
using Pixel16bfcC4 = Vector4<c_BFloat16>;

using Pixel16sC1  = Vector1<short>;
using Pixel16sC2  = Vector2<short>;
using Pixel16sC3  = Vector3<short>;
using Pixel16sC4  = Vector4<short>;
using Pixel16sC4A = Vector4A<short>;

using Pixel16scC1 = Vector1<c_short>;
using Pixel16scC2 = Vector2<c_short>;
using Pixel16scC3 = Vector3<c_short>;
using Pixel16scC4 = Vector4<c_short>;

using Pixel16uC1  = Vector1<ushort>;
using Pixel16uC2  = Vector2<ushort>;
using Pixel16uC3  = Vector3<ushort>;
using Pixel16uC4  = Vector4<ushort>;
using Pixel16uC4A = Vector4A<ushort>;

using Pixel8sC1  = Vector1<sbyte>;
using Pixel8sC2  = Vector2<sbyte>;
using Pixel8sC3  = Vector3<sbyte>;
using Pixel8sC4  = Vector4<sbyte>;
using Pixel8sC4A = Vector4A<sbyte>;

using Pixel8uC1  = Vector1<byte>;
using Pixel8uC2  = Vector2<byte>;
using Pixel8uC3  = Vector3<byte>;
using Pixel8uC4  = Vector4<byte>;
using Pixel8uC4A = Vector4A<byte>;
#pragma endregion

#pragma region PixelType enum
/// <summary>
/// Enumeration of all supported pixel types in OPP
/// </summary>
enum class PixelTypeEnum : byte
{
    // undefined or unsupported
    Unknown,
    // double
    PTE64fC1,
    PTE64fC2,
    PTE64fC3,
    PTE64fC4,
    PTE64fC4A,
    // double complex
    PTE64fcC1,
    PTE64fcC2,
    PTE64fcC3,
    PTE64fcC4,
    // float
    PTE32fC1,
    PTE32fC2,
    PTE32fC3,
    PTE32fC4,
    PTE32fC4A,
    // complex float
    PTE32fcC1,
    PTE32fcC2,
    PTE32fcC3,
    PTE32fcC4,
    // float16
    PTE16fC1,
    PTE16fC2,
    PTE16fC3,
    PTE16fC4,
    PTE16fC4A,
    // float16 complex
    PTE16fcC1,
    PTE16fcC2,
    PTE16fcC3,
    PTE16fcC4,
    // bfloat16
    PTE16bfC1,
    PTE16bfC2,
    PTE16bfC3,
    PTE16bfC4,
    PTE16bfC4A,
    // bfloat16 complex
    PTE16bfcC1,
    PTE16bfcC2,
    PTE16bfcC3,
    PTE16bfcC4,
    // int
    PTE32sC1,
    PTE32sC2,
    PTE32sC3,
    PTE32sC4,
    PTE32sC4A,
    // complex int
    PTE32scC1,
    PTE32scC2,
    PTE32scC3,
    PTE32scC4,
    // unsigned int
    PTE32uC1,
    PTE32uC2,
    PTE32uC3,
    PTE32uC4,
    PTE32uC4A,
    // short
    PTE16sC1,
    PTE16sC2,
    PTE16sC3,
    PTE16sC4,
    PTE16sC4A,
    // complex short
    PTE16scC1,
    PTE16scC2,
    PTE16scC3,
    PTE16scC4,
    // unsigned short
    PTE16uC1,
    PTE16uC2,
    PTE16uC3,
    PTE16uC4,
    PTE16uC4A,
    // signed byte
    PTE8sC1,
    PTE8sC2,
    PTE8sC3,
    PTE8sC4,
    PTE8sC4A,
    // unsigned byte
    PTE8uC1,
    PTE8uC2,
    PTE8uC3,
    PTE8uC4,
    PTE8uC4A
};

std::ostream &operator<<(std::ostream &aOs, const PixelTypeEnum &aPixelType);
std::wostream &operator<<(std::wostream &aOs, const PixelTypeEnum &aPixelType);

constexpr size_t GetChannelCount(PixelTypeEnum aPixelType)
{
    switch (aPixelType)
    {
        case opp::image::PixelTypeEnum::Unknown: // treat as single channel
        case opp::image::PixelTypeEnum::PTE64fC1:
        case opp::image::PixelTypeEnum::PTE64fcC1:
        case opp::image::PixelTypeEnum::PTE32fC1:
        case opp::image::PixelTypeEnum::PTE32fcC1:
        case opp::image::PixelTypeEnum::PTE16fC1:
        case opp::image::PixelTypeEnum::PTE16fcC1:
        case opp::image::PixelTypeEnum::PTE16bfC1:
        case opp::image::PixelTypeEnum::PTE16bfcC1:
        case opp::image::PixelTypeEnum::PTE32sC1:
        case opp::image::PixelTypeEnum::PTE32scC1:
        case opp::image::PixelTypeEnum::PTE32uC1:
        case opp::image::PixelTypeEnum::PTE16sC1:
        case opp::image::PixelTypeEnum::PTE16scC1:
        case opp::image::PixelTypeEnum::PTE16uC1:
        case opp::image::PixelTypeEnum::PTE8sC1:
        case opp::image::PixelTypeEnum::PTE8uC1:
            return 1;
        case opp::image::PixelTypeEnum::PTE64fC2:
        case opp::image::PixelTypeEnum::PTE64fcC2:
        case opp::image::PixelTypeEnum::PTE32fC2:
        case opp::image::PixelTypeEnum::PTE32fcC2:
        case opp::image::PixelTypeEnum::PTE16fC2:
        case opp::image::PixelTypeEnum::PTE16fcC2:
        case opp::image::PixelTypeEnum::PTE16bfC2:
        case opp::image::PixelTypeEnum::PTE16bfcC2:
        case opp::image::PixelTypeEnum::PTE32sC2:
        case opp::image::PixelTypeEnum::PTE32scC2:
        case opp::image::PixelTypeEnum::PTE32uC2:
        case opp::image::PixelTypeEnum::PTE16sC2:
        case opp::image::PixelTypeEnum::PTE16scC2:
        case opp::image::PixelTypeEnum::PTE16uC2:
        case opp::image::PixelTypeEnum::PTE8sC2:
        case opp::image::PixelTypeEnum::PTE8uC2:
            return 2;
        case opp::image::PixelTypeEnum::PTE64fC3:
        case opp::image::PixelTypeEnum::PTE64fcC3:
        case opp::image::PixelTypeEnum::PTE32fC3:
        case opp::image::PixelTypeEnum::PTE32fcC3:
        case opp::image::PixelTypeEnum::PTE16fC3:
        case opp::image::PixelTypeEnum::PTE16fcC3:
        case opp::image::PixelTypeEnum::PTE16bfC3:
        case opp::image::PixelTypeEnum::PTE16bfcC3:
        case opp::image::PixelTypeEnum::PTE32sC3:
        case opp::image::PixelTypeEnum::PTE32scC3:
        case opp::image::PixelTypeEnum::PTE32uC3:
        case opp::image::PixelTypeEnum::PTE16sC3:
        case opp::image::PixelTypeEnum::PTE16scC3:
        case opp::image::PixelTypeEnum::PTE16uC3:
        case opp::image::PixelTypeEnum::PTE8sC3:
        case opp::image::PixelTypeEnum::PTE8uC3:
            return 3;
        case opp::image::PixelTypeEnum::PTE64fC4:
        case opp::image::PixelTypeEnum::PTE64fcC4:
        case opp::image::PixelTypeEnum::PTE64fC4A:
        case opp::image::PixelTypeEnum::PTE32fC4:
        case opp::image::PixelTypeEnum::PTE32fC4A:
        case opp::image::PixelTypeEnum::PTE32fcC4:
        case opp::image::PixelTypeEnum::PTE16fC4:
        case opp::image::PixelTypeEnum::PTE16fcC4:
        case opp::image::PixelTypeEnum::PTE16fC4A:
        case opp::image::PixelTypeEnum::PTE16bfC4:
        case opp::image::PixelTypeEnum::PTE16bfcC4:
        case opp::image::PixelTypeEnum::PTE16bfC4A:
        case opp::image::PixelTypeEnum::PTE32sC4:
        case opp::image::PixelTypeEnum::PTE32sC4A:
        case opp::image::PixelTypeEnum::PTE32scC4:
        case opp::image::PixelTypeEnum::PTE32uC4:
        case opp::image::PixelTypeEnum::PTE32uC4A:
        case opp::image::PixelTypeEnum::PTE16sC4:
        case opp::image::PixelTypeEnum::PTE16sC4A:
        case opp::image::PixelTypeEnum::PTE16scC4:
        case opp::image::PixelTypeEnum::PTE16uC4:
        case opp::image::PixelTypeEnum::PTE16uC4A:
        case opp::image::PixelTypeEnum::PTE8sC4:
        case opp::image::PixelTypeEnum::PTE8sC4A:
        case opp::image::PixelTypeEnum::PTE8uC4:
        case opp::image::PixelTypeEnum::PTE8uC4A:
            return 4;
        default:
            return 0; // we should never get here...
    }
}

constexpr size_t GetPixelSizeInBytes(PixelTypeEnum aPixelType)
{
    switch (aPixelType)
    {
        case opp::image::PixelTypeEnum::Unknown:
            return 1;
        case opp::image::PixelTypeEnum::PTE64fcC4: // 64
            return 2ULL * 4ULL * 8ULL;
        case opp::image::PixelTypeEnum::PTE64fcC3: // 48
            return 3ULL * 16ULL;
        case opp::image::PixelTypeEnum::PTE64fcC2: // 32
        case opp::image::PixelTypeEnum::PTE64fC4:  // 32
        case opp::image::PixelTypeEnum::PTE64fC4A: // 32
        case opp::image::PixelTypeEnum::PTE32fcC4: // 32
        case opp::image::PixelTypeEnum::PTE32scC4: // 32
            return 4ULL * 8ULL;
        case opp::image::PixelTypeEnum::PTE64fC3:  // 24
        case opp::image::PixelTypeEnum::PTE32fcC3: // 24
        case opp::image::PixelTypeEnum::PTE32scC3: // 24
            return 3ULL * 8ULL;
        case opp::image::PixelTypeEnum::PTE64fcC1:  // 16
        case opp::image::PixelTypeEnum::PTE64fC2:   // 16
        case opp::image::PixelTypeEnum::PTE32fcC2:  // 16
        case opp::image::PixelTypeEnum::PTE32scC2:  // 16
        case opp::image::PixelTypeEnum::PTE32fC4:   // 16
        case opp::image::PixelTypeEnum::PTE32fC4A:  // 16
        case opp::image::PixelTypeEnum::PTE32sC4:   // 16
        case opp::image::PixelTypeEnum::PTE32sC4A:  // 16
        case opp::image::PixelTypeEnum::PTE32uC4:   // 16
        case opp::image::PixelTypeEnum::PTE32uC4A:  // 16
        case opp::image::PixelTypeEnum::PTE16fcC4:  // 16
        case opp::image::PixelTypeEnum::PTE16bfcC4: // 16
        case opp::image::PixelTypeEnum::PTE16scC4:  // 16
            return 2ULL * 8ULL;
        case opp::image::PixelTypeEnum::PTE32fC3:   // 12
        case opp::image::PixelTypeEnum::PTE32sC3:   // 12
        case opp::image::PixelTypeEnum::PTE32uC3:   // 12
        case opp::image::PixelTypeEnum::PTE16fcC3:  // 12
        case opp::image::PixelTypeEnum::PTE16bfcC3: // 12
        case opp::image::PixelTypeEnum::PTE16scC3:  // 12
            return 3ULL * 4ULL;
        case opp::image::PixelTypeEnum::PTE64fC1:   // 8
        case opp::image::PixelTypeEnum::PTE32fC2:   // 8
        case opp::image::PixelTypeEnum::PTE32fcC1:  // 8
        case opp::image::PixelTypeEnum::PTE32scC1:  // 8
        case opp::image::PixelTypeEnum::PTE32sC2:   // 8
        case opp::image::PixelTypeEnum::PTE32uC2:   // 8
        case opp::image::PixelTypeEnum::PTE16fcC2:  // 8
        case opp::image::PixelTypeEnum::PTE16bfcC2: // 8
        case opp::image::PixelTypeEnum::PTE16scC2:  // 8
        case opp::image::PixelTypeEnum::PTE16sC4:   // 8
        case opp::image::PixelTypeEnum::PTE16sC4A:  // 8
        case opp::image::PixelTypeEnum::PTE16uC4:   // 8
        case opp::image::PixelTypeEnum::PTE16uC4A:  // 8
        case opp::image::PixelTypeEnum::PTE16fC4:   // 8
        case opp::image::PixelTypeEnum::PTE16fC4A:  // 8
        case opp::image::PixelTypeEnum::PTE16bfC4:  // 8
        case opp::image::PixelTypeEnum::PTE16bfC4A: // 8
            return 8;
        case opp::image::PixelTypeEnum::PTE16sC3:  // 6
        case opp::image::PixelTypeEnum::PTE16uC3:  // 6
        case opp::image::PixelTypeEnum::PTE16fC3:  // 6
        case opp::image::PixelTypeEnum::PTE16bfC3: // 6
            return 6;
        case opp::image::PixelTypeEnum::PTE32fC1:   // 4
        case opp::image::PixelTypeEnum::PTE32sC1:   // 4
        case opp::image::PixelTypeEnum::PTE32uC1:   // 4
        case opp::image::PixelTypeEnum::PTE16fcC1:  // 4
        case opp::image::PixelTypeEnum::PTE16bfcC1: // 4
        case opp::image::PixelTypeEnum::PTE16scC1:  // 4
        case opp::image::PixelTypeEnum::PTE16sC2:   // 4
        case opp::image::PixelTypeEnum::PTE16uC2:   // 4
        case opp::image::PixelTypeEnum::PTE16fC2:   // 4
        case opp::image::PixelTypeEnum::PTE16bfC2:  // 4
        case opp::image::PixelTypeEnum::PTE8sC4:    // 4
        case opp::image::PixelTypeEnum::PTE8sC4A:   // 4
        case opp::image::PixelTypeEnum::PTE8uC4:    // 4
        case opp::image::PixelTypeEnum::PTE8uC4A:   // 4
            return 4;
        case opp::image::PixelTypeEnum::PTE8sC3: // 3
        case opp::image::PixelTypeEnum::PTE8uC3: // 3
            return 3;
        case opp::image::PixelTypeEnum::PTE16sC1:  // 2
        case opp::image::PixelTypeEnum::PTE16uC1:  // 2
        case opp::image::PixelTypeEnum::PTE16fC1:  // 2
        case opp::image::PixelTypeEnum::PTE16bfC1: // 2
        case opp::image::PixelTypeEnum::PTE8sC2:   // 2
        case opp::image::PixelTypeEnum::PTE8uC2:   // 2
            return 2;
        case opp::image::PixelTypeEnum::PTE8sC1: // 1
        case opp::image::PixelTypeEnum::PTE8uC1: // 1
            return 1;
        default:
            return 0; // we should never get here...
    }
}

template <typename T> struct pixel_type_name
{
    static constexpr char value[] = "Unknown";
};
// 64f
template <> struct pixel_type_name<Pixel64fC1>
{
    static constexpr char value[] = "Pixel64fC1";
};
template <> struct pixel_type_name<Pixel64fC2>
{
    static constexpr char value[] = "Pixel64fC2";
};
template <> struct pixel_type_name<Pixel64fC3>
{
    static constexpr char value[] = "Pixel64fC3";
};
template <> struct pixel_type_name<Pixel64fC4>
{
    static constexpr char value[] = "Pixel64fC4";
};
template <> struct pixel_type_name<Pixel64fC4A>
{
    static constexpr char value[] = "Pixel64fC4A";
};

// 64f complex
template <> struct pixel_type_name<Pixel64fcC1>
{
    static constexpr char value[] = "Pixel64fcC1";
};
template <> struct pixel_type_name<Pixel64fcC2>
{
    static constexpr char value[] = "Pixel64fcC2";
};
template <> struct pixel_type_name<Pixel64fcC3>
{
    static constexpr char value[] = "Pixel64fcC3";
};
template <> struct pixel_type_name<Pixel64fcC4>
{
    static constexpr char value[] = "Pixel64fcC4";
};

// 32f
template <> struct pixel_type_name<Pixel32fC1>
{
    static constexpr char value[] = "Pixel32fC1";
};
template <> struct pixel_type_name<Pixel32fC2>
{
    static constexpr char value[] = "Pixel32fC2";
};
template <> struct pixel_type_name<Pixel32fC3>
{
    static constexpr char value[] = "Pixel32fC3";
};
template <> struct pixel_type_name<Pixel32fC4>
{
    static constexpr char value[] = "Pixel32fC4";
};
template <> struct pixel_type_name<Pixel32fC4A>
{
    static constexpr char value[] = "Pixel32fC4A";
};

// 32fc
template <> struct pixel_type_name<Pixel32fcC1>
{
    static constexpr char value[] = "Pixel32fcC1";
};
template <> struct pixel_type_name<Pixel32fcC2>
{
    static constexpr char value[] = "Pixel32fcC2";
};
template <> struct pixel_type_name<Pixel32fcC3>
{
    static constexpr char value[] = "Pixel32fcC3";
};
template <> struct pixel_type_name<Pixel32fcC4>
{
    static constexpr char value[] = "Pixel32fcC4";
};

// 16f
template <> struct pixel_type_name<Pixel16fC1>
{
    static constexpr char value[] = "Pixel16fC1";
};
template <> struct pixel_type_name<Pixel16fC2>
{
    static constexpr char value[] = "Pixel16fC2";
};
template <> struct pixel_type_name<Pixel16fC3>
{
    static constexpr char value[] = "Pixel16fC3";
};
template <> struct pixel_type_name<Pixel16fC4>
{
    static constexpr char value[] = "Pixel16fC4";
};
template <> struct pixel_type_name<Pixel16fC4A>
{
    static constexpr char value[] = "Pixel16fC4A";
};

// 16f complex
template <> struct pixel_type_name<Pixel16fcC1>
{
    static constexpr char value[] = "Pixel16fcC1";
};
template <> struct pixel_type_name<Pixel16fcC2>
{
    static constexpr char value[] = "Pixel16fcC2";
};
template <> struct pixel_type_name<Pixel16fcC3>
{
    static constexpr char value[] = "Pixel16fcC3";
};
template <> struct pixel_type_name<Pixel16fcC4>
{
    static constexpr char value[] = "Pixel16fcC4";
};

// 16bf
template <> struct pixel_type_name<Pixel16bfC1>
{
    static constexpr char value[] = "Pixel16bfC1";
};
template <> struct pixel_type_name<Pixel16bfC2>
{
    static constexpr char value[] = "Pixel16bfC2";
};
template <> struct pixel_type_name<Pixel16bfC3>
{
    static constexpr char value[] = "Pixel16bfC3";
};
template <> struct pixel_type_name<Pixel16bfC4>
{
    static constexpr char value[] = "Pixel16bfC4";
};
template <> struct pixel_type_name<Pixel16bfC4A>
{
    static constexpr char value[] = "Pixel16bfC4A";
};

// 16bf complex
template <> struct pixel_type_name<Pixel16bfcC1>
{
    static constexpr char value[] = "Pixel16bfcC1";
};
template <> struct pixel_type_name<Pixel16bfcC2>
{
    static constexpr char value[] = "Pixel16bfcC2";
};
template <> struct pixel_type_name<Pixel16bfcC3>
{
    static constexpr char value[] = "Pixel16bfcC3";
};
template <> struct pixel_type_name<Pixel16bfcC4>
{
    static constexpr char value[] = "Pixel16bfcC4";
};

// 32s
template <> struct pixel_type_name<Pixel32sC1>
{
    static constexpr char value[] = "Pixel32sC1";
};
template <> struct pixel_type_name<Pixel32sC2>
{
    static constexpr char value[] = "Pixel32sC2";
};
template <> struct pixel_type_name<Pixel32sC3>
{
    static constexpr char value[] = "Pixel32sC3";
};
template <> struct pixel_type_name<Pixel32sC4>
{
    static constexpr char value[] = "Pixel32sC4";
};
template <> struct pixel_type_name<Pixel32sC4A>
{
    static constexpr char value[] = "Pixel32sC4A";
};

// 32sc
template <> struct pixel_type_name<Pixel32scC1>
{
    static constexpr char value[] = "Pixel32scC1";
};
template <> struct pixel_type_name<Pixel32scC2>
{
    static constexpr char value[] = "Pixel32scC2";
};
template <> struct pixel_type_name<Pixel32scC3>
{
    static constexpr char value[] = "Pixel32scC3";
};
template <> struct pixel_type_name<Pixel32scC4>
{
    static constexpr char value[] = "Pixel32scC4";
};

// 32u
template <> struct pixel_type_name<Pixel32uC1>
{
    static constexpr char value[] = "Pixel32uC1";
};
template <> struct pixel_type_name<Pixel32uC2>
{
    static constexpr char value[] = "Pixel32uC2";
};
template <> struct pixel_type_name<Pixel32uC3>
{
    static constexpr char value[] = "Pixel32uC3";
};
template <> struct pixel_type_name<Pixel32uC4>
{
    static constexpr char value[] = "Pixel32uC4";
};
template <> struct pixel_type_name<Pixel32uC4A>
{
    static constexpr char value[] = "Pixel32uC4A";
};

// 16s
template <> struct pixel_type_name<Pixel16sC1>
{
    static constexpr char value[] = "Pixel16sC1";
};
template <> struct pixel_type_name<Pixel16sC2>
{
    static constexpr char value[] = "Pixel16sC2";
};
template <> struct pixel_type_name<Pixel16sC3>
{
    static constexpr char value[] = "Pixel16sC3";
};
template <> struct pixel_type_name<Pixel16sC4>
{
    static constexpr char value[] = "Pixel16sC4";
};
template <> struct pixel_type_name<Pixel16sC4A>
{
    static constexpr char value[] = "Pixel16sC4A";
};

// 16sc
template <> struct pixel_type_name<Pixel16scC1>
{
    static constexpr char value[] = "Pixel16scC1";
};
template <> struct pixel_type_name<Pixel16scC2>
{
    static constexpr char value[] = "Pixel16scC2";
};
template <> struct pixel_type_name<Pixel16scC3>
{
    static constexpr char value[] = "Pixel16scC3";
};
template <> struct pixel_type_name<Pixel16scC4>
{
    static constexpr char value[] = "Pixel16scC4";
};

// 16u
template <> struct pixel_type_name<Pixel16uC1>
{
    static constexpr char value[] = "Pixel16uC1";
};
template <> struct pixel_type_name<Pixel16uC2>
{
    static constexpr char value[] = "Pixel16uC2";
};
template <> struct pixel_type_name<Pixel16uC3>
{
    static constexpr char value[] = "Pixel16uC3";
};
template <> struct pixel_type_name<Pixel16uC4>
{
    static constexpr char value[] = "Pixel16uC4";
};
template <> struct pixel_type_name<Pixel16uC4A>
{
    static constexpr char value[] = "Pixel16uC4A";
};

// 8s
template <> struct pixel_type_name<Pixel8sC1>
{
    static constexpr char value[] = "Pixel8sC1";
};
template <> struct pixel_type_name<Pixel8sC2>
{
    static constexpr char value[] = "Pixel8sC2";
};
template <> struct pixel_type_name<Pixel8sC3>
{
    static constexpr char value[] = "Pixel8sC3";
};
template <> struct pixel_type_name<Pixel8sC4>
{
    static constexpr char value[] = "Pixel8sC4";
};
template <> struct pixel_type_name<Pixel8sC4A>
{
    static constexpr char value[] = "Pixel8sC4A";
};

// 8u
template <> struct pixel_type_name<Pixel8uC1>
{
    static constexpr char value[] = "Pixel8uC1";
};
template <> struct pixel_type_name<Pixel8uC2>
{
    static constexpr char value[] = "Pixel8uC2";
};
template <> struct pixel_type_name<Pixel8uC3>
{
    static constexpr char value[] = "Pixel8uC3";
};
template <> struct pixel_type_name<Pixel8uC4>
{
    static constexpr char value[] = "Pixel8uC4";
};
template <> struct pixel_type_name<Pixel8uC4A>
{
    static constexpr char value[] = "Pixel8uC4A";
};

// template <VectorOrComplexType T> inline constexpr char pixel_type_name_v[] = pixel_type_name<T>::value;
#pragma endregion

#pragma region PixelType type traits
template <typename T> struct channel_count : vector_size<T>
{
};

template <typename T> struct pixel_basetype : remove_vector<T>
{
};

template <typename T> using pixel_basetype_t = typename pixel_basetype<T>::type;

template <typename T>
concept PixelType = channel_count<T>::value > 0 && channel_count<T>::value <= 4;

template <typename T> struct default_compute_type_for
{
    using type = void;
};
// 64f
template <> struct default_compute_type_for<Pixel64fC1>
{
    using type = Pixel64fC1;
};
template <> struct default_compute_type_for<Pixel64fC2>
{
    using type = Pixel64fC2;
};
template <> struct default_compute_type_for<Pixel64fC3>
{
    using type = Pixel64fC3;
};
template <> struct default_compute_type_for<Pixel64fC4>
{
    using type = Pixel64fC4;
};
template <> struct default_compute_type_for<Pixel64fC4A>
{
    using type = Pixel64fC4A;
};

// 64f
template <> struct default_compute_type_for<Pixel64fcC1>
{
    using type = Pixel64fcC1;
};
template <> struct default_compute_type_for<Pixel64fcC2>
{
    using type = Pixel64fcC2;
};
template <> struct default_compute_type_for<Pixel64fcC3>
{
    using type = Pixel64fcC3;
};
template <> struct default_compute_type_for<Pixel64fcC4>
{
    using type = Pixel64fcC4;
};

// 32f
template <> struct default_compute_type_for<Pixel32fC1>
{
    using type = Pixel32fC1;
};
template <> struct default_compute_type_for<Pixel32fC2>
{
    using type = Pixel32fC2;
};
template <> struct default_compute_type_for<Pixel32fC3>
{
    using type = Pixel32fC3;
};
template <> struct default_compute_type_for<Pixel32fC4>
{
    using type = Pixel32fC4;
};
template <> struct default_compute_type_for<Pixel32fC4A>
{
    using type = Pixel32fC4A;
};

// 32fc
template <> struct default_compute_type_for<Pixel32fcC1>
{
    using type = Pixel32fcC1;
};
template <> struct default_compute_type_for<Pixel32fcC2>
{
    using type = Pixel32fcC2;
};
template <> struct default_compute_type_for<Pixel32fcC3>
{
    using type = Pixel32fcC3;
};
template <> struct default_compute_type_for<Pixel32fcC4>
{
    using type = Pixel32fcC4;
};

// 16f
template <> struct default_compute_type_for<Pixel16fC1>
{
    using type = Pixel32fC1;
};
template <> struct default_compute_type_for<Pixel16fC2>
{
    using type = Pixel32fC2;
};
template <> struct default_compute_type_for<Pixel16fC3>
{
    using type = Pixel32fC3;
};
template <> struct default_compute_type_for<Pixel16fC4>
{
    using type = Pixel32fC4;
};
template <> struct default_compute_type_for<Pixel16fC4A>
{
    using type = Pixel32fC4A;
};

// 16fc
template <> struct default_compute_type_for<Pixel16fcC1>
{
    using type = Pixel32fcC1;
};
template <> struct default_compute_type_for<Pixel16fcC2>
{
    using type = Pixel32fcC2;
};
template <> struct default_compute_type_for<Pixel16fcC3>
{
    using type = Pixel32fcC3;
};
template <> struct default_compute_type_for<Pixel16fcC4>
{
    using type = Pixel32fcC4;
};

// 16bf
template <> struct default_compute_type_for<Pixel16bfC1>
{
    using type = Pixel32fC1;
};
template <> struct default_compute_type_for<Pixel16bfC2>
{
    using type = Pixel32fC2;
};
template <> struct default_compute_type_for<Pixel16bfC3>
{
    using type = Pixel32fC3;
};
template <> struct default_compute_type_for<Pixel16bfC4>
{
    using type = Pixel32fC4;
};
template <> struct default_compute_type_for<Pixel16bfC4A>
{
    using type = Pixel32fC4A;
};

// 16bfc
template <> struct default_compute_type_for<Pixel16bfcC1>
{
    using type = Pixel32fcC1;
};
template <> struct default_compute_type_for<Pixel16bfcC2>
{
    using type = Pixel32fcC2;
};
template <> struct default_compute_type_for<Pixel16bfcC3>
{
    using type = Pixel32fcC3;
};
template <> struct default_compute_type_for<Pixel16bfcC4>
{
    using type = Pixel32fcC4;
};

// 32s -> 64f
template <> struct default_compute_type_for<Pixel32sC1>
{
    using type = Pixel64fC1;
};
template <> struct default_compute_type_for<Pixel32sC2>
{
    using type = Pixel64fC2;
};
template <> struct default_compute_type_for<Pixel32sC3>
{
    using type = Pixel64fC3;
};
template <> struct default_compute_type_for<Pixel32sC4>
{
    using type = Pixel64fC4;
};
template <> struct default_compute_type_for<Pixel32sC4A>
{
    using type = Pixel64fC4A;
};

// 32sc -> 32fc
template <> struct default_compute_type_for<Pixel32scC1>
{
    using type = Pixel32fcC1;
};
template <> struct default_compute_type_for<Pixel32scC2>
{
    using type = Pixel32fcC2;
};
template <> struct default_compute_type_for<Pixel32scC3>
{
    using type = Pixel32fcC3;
};
template <> struct default_compute_type_for<Pixel32scC4>
{
    using type = Pixel32fcC4;
};

// 32u -> 64f
template <> struct default_compute_type_for<Pixel32uC1>
{
    using type = Pixel64fC1;
};
template <> struct default_compute_type_for<Pixel32uC2>
{
    using type = Pixel64fC2;
};
template <> struct default_compute_type_for<Pixel32uC3>
{
    using type = Pixel64fC3;
};
template <> struct default_compute_type_for<Pixel32uC4>
{
    using type = Pixel64fC4;
};
template <> struct default_compute_type_for<Pixel32uC4A>
{
    using type = Pixel64fC4A;
};

// 16s -> 32f
template <> struct default_compute_type_for<Pixel16sC1>
{
    using type = Pixel32fC1;
};
template <> struct default_compute_type_for<Pixel16sC2>
{
    using type = Pixel32fC2;
};
template <> struct default_compute_type_for<Pixel16sC3>
{
    using type = Pixel32fC3;
};
template <> struct default_compute_type_for<Pixel16sC4>
{
    using type = Pixel32fC4;
};
template <> struct default_compute_type_for<Pixel16sC4A>
{
    using type = Pixel32fC4A;
};

// 16sc -> 32fc
template <> struct default_compute_type_for<Pixel16scC1>
{
    using type = Pixel32fcC1;
};
template <> struct default_compute_type_for<Pixel16scC2>
{
    using type = Pixel32fcC2;
};
template <> struct default_compute_type_for<Pixel16scC3>
{
    using type = Pixel32fcC3;
};
template <> struct default_compute_type_for<Pixel16scC4>
{
    using type = Pixel32fcC4;
};

// 16u -> 32f
template <> struct default_compute_type_for<Pixel16uC1>
{
    using type = Pixel32fC1;
};
template <> struct default_compute_type_for<Pixel16uC2>
{
    using type = Pixel32fC2;
};
template <> struct default_compute_type_for<Pixel16uC3>
{
    using type = Pixel32fC3;
};
template <> struct default_compute_type_for<Pixel16uC4>
{
    using type = Pixel32fC4;
};
template <> struct default_compute_type_for<Pixel16uC4A>
{
    using type = Pixel32fC4A;
};

// 8s -> 32f
template <> struct default_compute_type_for<Pixel8sC1>
{
    using type = Pixel32fC1;
};
template <> struct default_compute_type_for<Pixel8sC2>
{
    using type = Pixel32fC2;
};
template <> struct default_compute_type_for<Pixel8sC3>
{
    using type = Pixel32fC3;
};
template <> struct default_compute_type_for<Pixel8sC4>
{
    using type = Pixel32fC4;
};
template <> struct default_compute_type_for<Pixel8sC4A>
{
    using type = Pixel32fC4A;
};

// 8u -> 32f
template <> struct default_compute_type_for<Pixel8uC1>
{
    using type = Pixel32fC1;
};
template <> struct default_compute_type_for<Pixel8uC2>
{
    using type = Pixel32fC2;
};
template <> struct default_compute_type_for<Pixel8uC3>
{
    using type = Pixel32fC3;
};
template <> struct default_compute_type_for<Pixel8uC4>
{
    using type = Pixel32fC4;
};
template <> struct default_compute_type_for<Pixel8uC4A>
{
    using type = Pixel32fC4A;
};

template <typename T> using default_compute_type_for_t = typename default_compute_type_for<T>::type;

// returns true if T is a Vector4A type with w as alpha channel
template <typename T> struct has_alpha_channel : std::false_type
{
};
template <typename T> struct has_alpha_channel<Vector4A<T>> : std::true_type
{
};
template <class T> inline constexpr bool has_alpha_channel_v = has_alpha_channel<T>::value;

// indicates if it is better to load then entire vector4A for the alpha channel or just the single alpha channel
template <typename T> struct load_full_vector_for_alpha : std::false_type
{
};
// vector fragments smaller than 32 bit are anyway fetched as a 32 bit word, no need to slice things
template <ByteSizeType T> struct load_full_vector_for_alpha<Vector4A<T>> : std::true_type
{
};
// vector fragments smaller than 32 bit are anyway fetched as a 32 bit word, no need to slice things
// this needs to be tested if it is better for 16 bit data types to load everything or just the alpha channel, or only z
// and w?
template <TwoBytesSizeType T> struct load_full_vector_for_alpha<Vector4A<T>> : std::true_type
{
};
template <class T> inline constexpr bool load_full_vector_for_alpha_v = load_full_vector_for_alpha<T>::value;

#pragma region Link the type PixelType with the enum PixelType
template <PixelTypeEnum pixelType> struct pixel_type
{
    using type = void;
};
// 64f
template <> struct pixel_type<PixelTypeEnum::PTE64fC1>
{
    using type = Pixel64fC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE64fC2>
{
    using type = Pixel64fC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE64fC3>
{
    using type = Pixel64fC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE64fC4>
{
    using type = Pixel64fC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE64fC4A>
{
    using type = Pixel64fC4A;
};

// 64f complex
template <> struct pixel_type<PixelTypeEnum::PTE64fcC1>
{
    using type = Pixel64fcC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE64fcC2>
{
    using type = Pixel64fcC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE64fcC3>
{
    using type = Pixel64fcC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE64fcC4>
{
    using type = Pixel64fcC4;
};

// 32f
template <> struct pixel_type<PixelTypeEnum::PTE32fC1>
{
    using type = Pixel32fC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE32fC2>
{
    using type = Pixel32fC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE32fC3>
{
    using type = Pixel32fC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE32fC4>
{
    using type = Pixel32fC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE32fC4A>
{
    using type = Pixel32fC4A;
};

// 32fc
template <> struct pixel_type<PixelTypeEnum::PTE32fcC1>
{
    using type = Pixel32fcC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE32fcC2>
{
    using type = Pixel32fcC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE32fcC3>
{
    using type = Pixel32fcC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE32fcC4>
{
    using type = Pixel32fcC4;
};

// 16f
template <> struct pixel_type<PixelTypeEnum::PTE16fC1>
{
    using type = Pixel16fC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE16fC2>
{
    using type = Pixel16fC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE16fC3>
{
    using type = Pixel16fC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE16fC4>
{
    using type = Pixel16fC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE16fC4A>
{
    using type = Pixel16fC4A;
};

// 16f
template <> struct pixel_type<PixelTypeEnum::PTE16fcC1>
{
    using type = Pixel16fcC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE16fcC2>
{
    using type = Pixel16fcC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE16fcC3>
{
    using type = Pixel16fcC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE16fcC4>
{
    using type = Pixel16fcC4;
};

// 16bf
template <> struct pixel_type<PixelTypeEnum::PTE16bfC1>
{
    using type = Pixel16bfC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE16bfC2>
{
    using type = Pixel16bfC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE16bfC3>
{
    using type = Pixel16bfC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE16bfC4>
{
    using type = Pixel16bfC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE16bfC4A>
{
    using type = Pixel16bfC4A;
};

// 16bf
template <> struct pixel_type<PixelTypeEnum::PTE16bfcC1>
{
    using type = Pixel16bfcC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE16bfcC2>
{
    using type = Pixel16bfcC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE16bfcC3>
{
    using type = Pixel16bfcC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE16bfcC4>
{
    using type = Pixel16bfcC4;
};

// 32s
template <> struct pixel_type<PixelTypeEnum::PTE32sC1>
{
    using type = Pixel32sC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE32sC2>
{
    using type = Pixel32sC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE32sC3>
{
    using type = Pixel32sC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE32sC4>
{
    using type = Pixel32sC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE32sC4A>
{
    using type = Pixel32sC4A;
};

// 32sc
template <> struct pixel_type<PixelTypeEnum::PTE32scC1>
{
    using type = Pixel32scC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE32scC2>
{
    using type = Pixel32scC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE32scC3>
{
    using type = Pixel32scC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE32scC4>
{
    using type = Pixel32scC4;
};

// 32u
template <> struct pixel_type<PixelTypeEnum::PTE32uC1>
{
    using type = Pixel32uC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE32uC2>
{
    using type = Pixel32uC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE32uC3>
{
    using type = Pixel32uC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE32uC4>
{
    using type = Pixel32uC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE32uC4A>
{
    using type = Pixel32uC4A;
};

// 16s
template <> struct pixel_type<PixelTypeEnum::PTE16sC1>
{
    using type = Pixel16sC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE16sC2>
{
    using type = Pixel16sC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE16sC3>
{
    using type = Pixel16sC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE16sC4>
{
    using type = Pixel16sC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE16sC4A>
{
    using type = Pixel16sC4A;
};

// 16sc
template <> struct pixel_type<PixelTypeEnum::PTE16scC1>
{
    using type = Pixel16scC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE16scC2>
{
    using type = Pixel16scC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE16scC3>
{
    using type = Pixel16scC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE16scC4>
{
    using type = Pixel16scC4;
};

// 16u
template <> struct pixel_type<PixelTypeEnum::PTE16uC1>
{
    using type = Pixel16uC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE16uC2>
{
    using type = Pixel16uC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE16uC3>
{
    using type = Pixel16uC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE16uC4>
{
    using type = Pixel16uC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE16uC4A>
{
    using type = Pixel16uC4A;
};

// 8s
template <> struct pixel_type<PixelTypeEnum::PTE8sC1>
{
    using type = Pixel8sC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE8sC2>
{
    using type = Pixel8sC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE8sC3>
{
    using type = Pixel8sC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE8sC4>
{
    using type = Pixel8sC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE8sC4A>
{
    using type = Pixel8sC4A;
};

// 8u
template <> struct pixel_type<PixelTypeEnum::PTE8uC1>
{
    using type = Pixel8uC1;
};
template <> struct pixel_type<PixelTypeEnum::PTE8uC2>
{
    using type = Pixel8uC2;
};
template <> struct pixel_type<PixelTypeEnum::PTE8uC3>
{
    using type = Pixel8uC3;
};
template <> struct pixel_type<PixelTypeEnum::PTE8uC4>
{
    using type = Pixel8uC4;
};
template <> struct pixel_type<PixelTypeEnum::PTE8uC4A>
{
    using type = Pixel8uC4A;
};

template <PixelTypeEnum pixelType> using pixel_type_t = typename pixel_type<pixelType>::type;

template <typename pixelT> struct pixel_type_enum
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::Unknown;
};
// 64f
template <> struct pixel_type_enum<Pixel64fC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE64fC1;
};
template <> struct pixel_type_enum<Pixel64fC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE64fC2;
};
template <> struct pixel_type_enum<Pixel64fC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE64fC3;
};
template <> struct pixel_type_enum<Pixel64fC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE64fC4;
};
template <> struct pixel_type_enum<Pixel64fC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE64fC4A;
};

// 64f complex
template <> struct pixel_type_enum<Pixel64fcC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE64fcC1;
};
template <> struct pixel_type_enum<Pixel64fcC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE64fcC2;
};
template <> struct pixel_type_enum<Pixel64fcC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE64fcC3;
};
template <> struct pixel_type_enum<Pixel64fcC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE64fcC4;
};

// 32f
template <> struct pixel_type_enum<Pixel32fC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32fC1;
};
template <> struct pixel_type_enum<Pixel32fC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32fC2;
};
template <> struct pixel_type_enum<Pixel32fC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32fC3;
};
template <> struct pixel_type_enum<Pixel32fC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32fC4;
};
template <> struct pixel_type_enum<Pixel32fC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32fC4A;
};

// 32fc
template <> struct pixel_type_enum<Pixel32fcC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32fcC1;
};
template <> struct pixel_type_enum<Pixel32fcC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32fcC2;
};
template <> struct pixel_type_enum<Pixel32fcC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32fcC3;
};
template <> struct pixel_type_enum<Pixel32fcC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32fcC4;
};

// 16f
template <> struct pixel_type_enum<Pixel16fC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16fC1;
};
template <> struct pixel_type_enum<Pixel16fC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16fC2;
};
template <> struct pixel_type_enum<Pixel16fC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16fC3;
};
template <> struct pixel_type_enum<Pixel16fC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16fC4;
};
template <> struct pixel_type_enum<Pixel16fC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16fC4A;
};

// 16fc
template <> struct pixel_type_enum<Pixel16fcC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16fcC1;
};
template <> struct pixel_type_enum<Pixel16fcC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16fcC2;
};
template <> struct pixel_type_enum<Pixel16fcC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16fcC3;
};
template <> struct pixel_type_enum<Pixel16fcC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16fcC4;
};

// 16bf
template <> struct pixel_type_enum<Pixel16bfC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16bfC1;
};
template <> struct pixel_type_enum<Pixel16bfC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16bfC2;
};
template <> struct pixel_type_enum<Pixel16bfC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16bfC3;
};
template <> struct pixel_type_enum<Pixel16bfC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16bfC4;
};
template <> struct pixel_type_enum<Pixel16bfC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16bfC4A;
};

// 16bfc
template <> struct pixel_type_enum<Pixel16bfcC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16bfcC1;
};
template <> struct pixel_type_enum<Pixel16bfcC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16bfcC2;
};
template <> struct pixel_type_enum<Pixel16bfcC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16bfcC3;
};
template <> struct pixel_type_enum<Pixel16bfcC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16bfcC4;
};

// 32s
template <> struct pixel_type_enum<Pixel32sC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32sC1;
};
template <> struct pixel_type_enum<Pixel32sC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32sC2;
};
template <> struct pixel_type_enum<Pixel32sC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32sC3;
};
template <> struct pixel_type_enum<Pixel32sC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32sC4;
};
template <> struct pixel_type_enum<Pixel32sC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32sC4A;
};

// 32sc
template <> struct pixel_type_enum<Pixel32scC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32scC1;
};
template <> struct pixel_type_enum<Pixel32scC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32scC2;
};
template <> struct pixel_type_enum<Pixel32scC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32scC3;
};
template <> struct pixel_type_enum<Pixel32scC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32scC4;
};

// 32u
template <> struct pixel_type_enum<Pixel32uC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32uC1;
};
template <> struct pixel_type_enum<Pixel32uC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32uC2;
};
template <> struct pixel_type_enum<Pixel32uC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32uC3;
};
template <> struct pixel_type_enum<Pixel32uC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32uC4;
};
template <> struct pixel_type_enum<Pixel32uC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE32uC4A;
};

// 16s
template <> struct pixel_type_enum<Pixel16sC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16sC1;
};
template <> struct pixel_type_enum<Pixel16sC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16sC2;
};
template <> struct pixel_type_enum<Pixel16sC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16sC3;
};
template <> struct pixel_type_enum<Pixel16sC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16sC4;
};
template <> struct pixel_type_enum<Pixel16sC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16sC4A;
};

// 16sc
template <> struct pixel_type_enum<Pixel16scC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16scC1;
};
template <> struct pixel_type_enum<Pixel16scC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16scC2;
};
template <> struct pixel_type_enum<Pixel16scC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16scC3;
};
template <> struct pixel_type_enum<Pixel16scC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16scC4;
};

// 16u
template <> struct pixel_type_enum<Pixel16uC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16uC1;
};
template <> struct pixel_type_enum<Pixel16uC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16uC2;
};
template <> struct pixel_type_enum<Pixel16uC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16uC3;
};
template <> struct pixel_type_enum<Pixel16uC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16uC4;
};
template <> struct pixel_type_enum<Pixel16uC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE16uC4A;
};

// 8s
template <> struct pixel_type_enum<Pixel8sC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8sC1;
};
template <> struct pixel_type_enum<Pixel8sC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8sC2;
};
template <> struct pixel_type_enum<Pixel8sC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8sC3;
};
template <> struct pixel_type_enum<Pixel8sC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8sC4;
};
template <> struct pixel_type_enum<Pixel8sC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8sC4A;
};

// 8u
template <> struct pixel_type_enum<Pixel8uC1>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8uC1;
};
template <> struct pixel_type_enum<Pixel8uC2>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8uC2;
};
template <> struct pixel_type_enum<Pixel8uC3>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8uC3;
};
template <> struct pixel_type_enum<Pixel8uC4>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8uC4;
};
template <> struct pixel_type_enum<Pixel8uC4A>
{
    static constexpr PixelTypeEnum pixelType = PixelTypeEnum::PTE8uC4A;
};

template <typename pixelT> using pixel_type_enum_t = pixel_type_enum<pixelT>::pixelType;
#pragma endregion
#pragma endregion

} // namespace opp::image