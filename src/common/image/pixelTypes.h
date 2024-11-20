#pragma once
#include <common/defines.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp::image
{
using Pixel64fC1 = Vector1<double>;
using Pixel64fC2 = Vector2<double>;
using Pixel64fC3 = Vector3<double>;
using Pixel64fC4 = Vector4<double>;

using Pixel32fC1 = Vector1<float>;
using Pixel32fC2 = Vector2<float>;
using Pixel32fC3 = Vector3<float>;
using Pixel32fC4 = Vector4<float>;

using Pixel32sC1 = Vector1<int>;
using Pixel32sC2 = Vector2<int>;
using Pixel32sC3 = Vector3<int>;
using Pixel32sC4 = Vector4<int>;

using Pixel32uC1 = Vector1<uint>;
using Pixel32uC2 = Vector2<uint>;
using Pixel32uC3 = Vector3<uint>;
using Pixel32uC4 = Vector4<uint>;

using Pixel16sC1 = Vector1<short>;
using Pixel16sC2 = Vector2<short>;
using Pixel16sC3 = Vector3<short>;
using Pixel16sC4 = Vector4<short>;

using Pixel16uC1 = Vector1<ushort>;
using Pixel16uC2 = Vector2<ushort>;
using Pixel16uC3 = Vector3<ushort>;
using Pixel16uC4 = Vector4<ushort>;

using Pixel82C1 = Vector1<sbyte>;
using Pixel82C2 = Vector2<sbyte>;
using Pixel82C3 = Vector3<sbyte>;
using Pixel82C4 = Vector4<sbyte>;

using Pixel8uC1 = Vector1<byte>;
using Pixel8uC2 = Vector2<byte>;
using Pixel8uC3 = Vector3<byte>;
using Pixel8uC4 = Vector4<byte>;

template <typename T> struct channel_count : vector_size<T>
{
};

template <typename T> struct pixel_basetype : remove_vector<T>
{
};

template <typename T>
concept PixelType = channel_count<T>::value > 0 && channel_count<T>::value <= 4;
} // namespace opp::image