#pragma once
#include <common/complex.h>
#include <common/complex_typetraits.h>
#include <common/defines.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp::image
{
using Pixel64fC1  = Vector1<double>;
using Pixel64fC2  = Vector2<double>;
using Pixel64fC3  = Vector3<double>;
using Pixel64fC4  = Vector4<double>;
using Pixel64fC4A = Vector4A<double>;

using Pixel32fC1  = Vector1<float>;
using Pixel32fC2  = Vector2<float>;
using Pixel32fC3  = Vector3<float>;
using Pixel32fC4  = Vector4<float>;
using Pixel32fC4A = Vector4A<float>;

using Pixel32fcC1 = Complex<float>;
using Pixel32fcC2 = Vector2<Complex<float>>;
using Pixel32fcC3 = Vector3<Complex<float>>;
using Pixel32fcC4 = Vector4<Complex<float>>;

using Pixel32sC1  = Vector1<int>;
using Pixel32sC2  = Vector2<int>;
using Pixel32sC3  = Vector3<int>;
using Pixel32sC4  = Vector4<int>;
using Pixel32sC4A = Vector4A<int>;

using Pixel32scC1 = Complex<int>;
using Pixel32scC2 = Vector2<Complex<int>>;
using Pixel32scC3 = Vector3<Complex<int>>;
using Pixel32scC4 = Vector4<Complex<int>>;

using Pixel32uC1  = Vector1<uint>;
using Pixel32uC2  = Vector2<uint>;
using Pixel32uC3  = Vector3<uint>;
using Pixel32uC4  = Vector4<uint>;
using Pixel32uC4A = Vector4A<uint>;

using Pixel16sC1  = Vector1<short>;
using Pixel16sC2  = Vector2<short>;
using Pixel16sC3  = Vector3<short>;
using Pixel16sC4  = Vector4<short>;
using Pixel16sC4A = Vector4A<short>;

using Pixel16scC1 = Complex<short>;
using Pixel16scC2 = Vector2<Complex<short>>;
using Pixel16scC3 = Vector3<Complex<short>>;
using Pixel16scC4 = Vector4<Complex<short>>;

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

template <typename T> struct channel_count : vector_size<T>
{
};

template <typename T> struct pixel_basetype : remove_vector<T>
{
};

// return T also for Vector<Complex<T>>
template <typename T> struct pixel_basetype<Vector1<Complex<T>>>
{
    using type = T;
};
template <typename T> struct pixel_basetype<Vector2<Complex<T>>>
{
    using type = T;
};
template <typename T> struct pixel_basetype<Vector3<Complex<T>>>
{
    using type = T;
};
template <typename T> struct pixel_basetype<Vector4<Complex<T>>>
{
    using type = T;
};

template <typename T> using pixel_basetype_t = typename pixel_basetype<T>::type;

template <typename T>
concept PixelType = channel_count<T>::value > 0 && channel_count<T>::value <= 4;

template <VectorOrComplexType T> struct default_compute_type_for
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
    using type = Pixel32scC1;
};
template <> struct default_compute_type_for<Pixel32scC2>
{
    using type = Pixel32scC2;
};
template <> struct default_compute_type_for<Pixel32scC3>
{
    using type = Pixel32scC3;
};
template <> struct default_compute_type_for<Pixel32scC4>
{
    using type = Pixel32scC4;
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
} // namespace opp::image