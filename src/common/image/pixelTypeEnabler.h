#pragma once
#include "pixelTypes.h"
#include <concepts>

// define concepts and constexpr based on the C-preprocessor macros to en- or disable OPP modules
namespace opp::image
{
template <typename T> struct enable_pixel_type : std::bool_constant<false>
{
};
template <class T> inline constexpr bool enable_pixel_type_v = enable_pixel_type<T>::value;

template <typename T>
concept oppEnablePixelType = enable_pixel_type_v<T>;

// if these macros are not set in CMAKE, fall back to enable:
#ifndef OPPi_ENABLE_DOUBLE_TYPE
#define OPPi_ENABLE_DOUBLE_TYPE 1
#endif
#ifndef OPPi_ENABLE_FLOAT_TYPE
#define OPPi_ENABLE_FLOAT_TYPE 1
#endif
#ifndef OPPi_ENABLE_HALFFLOAT16_TYPE
#define OPPi_ENABLE_HALFFLOAT16_TYPE 1
#endif
#ifndef OPPi_ENABLE_BFLOAT16_TYPE
#define OPPi_ENABLE_BFLOAT16_TYPE 1
#endif
#ifndef OPPi_ENABLE_INT32_TYPE
#define OPPi_ENABLE_INT32_TYPE 1
#endif
#ifndef OPPi_ENABLE_UINT32_TYPE
#define OPPi_ENABLE_UINT32_TYPE 1
#endif
#ifndef OPPi_ENABLE_INT16_TYPE
#define OPPi_ENABLE_INT16_TYPE 1
#endif
#ifndef OPPi_ENABLE_UINT16_TYPE
#define OPPi_ENABLE_UINT16_TYPE 1
#endif
#ifndef OPPi_ENABLE_INT8_TYPE
#define OPPi_ENABLE_INT8_TYPE 1
#endif
#ifndef OPPi_ENABLE_UINT8_TYPE
#define OPPi_ENABLE_UINT8_TYPE 1
#endif

#ifndef OPPi_ENABLE_COMPLEX_DOUBLE_TYPE
#define OPPi_ENABLE_COMPLEX_DOUBLE_TYPE 1
#endif
#ifndef OPPi_ENABLE_COMPLEX_FLOAT_TYPE
#define OPPi_ENABLE_COMPLEX_FLOAT_TYPE 1
#endif
#ifndef OPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE
#define OPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE 1
#endif
#ifndef OPPi_ENABLE_COMPLEX_BFLOAT16_TYPE
#define OPPi_ENABLE_COMPLEX_BFLOAT16_TYPE 1
#endif
#ifndef OPPi_ENABLE_COMPLEX_INT32_TYPE
#define OPPi_ENABLE_COMPLEX_INT32_TYPE 1
#endif
#ifndef OPPi_ENABLE_COMPLEX_INT16_TYPE
#define OPPi_ENABLE_COMPLEX_INT16_TYPE 1
#endif

#ifndef OPPi_ENABLE_ONE_CHANNEL
#define OPPi_ENABLE_ONE_CHANNEL 1
#endif
#ifndef OPPi_ENABLE_TWO_CHANNEL
#define OPPi_ENABLE_TWO_CHANNEL 1
#endif
#ifndef OPPi_ENABLE_THREE_CHANNEL
#define OPPi_ENABLE_THREE_CHANNEL 1
#endif
#ifndef OPPi_ENABLE_FOUR_CHANNEL
#define OPPi_ENABLE_FOUR_CHANNEL 1
#endif
#ifndef OPPi_ENABLE_FOURALPHA_CHANNEL
#define OPPi_ENABLE_FOURALPHA_CHANNEL 1
#endif

// double
#if (OPPi_ENABLE_DOUBLE_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel64fC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_DOUBLE_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel64fC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_DOUBLE_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel64fC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_DOUBLE_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel64fC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_DOUBLE_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel64fC4A> : std::bool_constant<true>
{
};
#endif

// float
#if (OPPi_ENABLE_FLOAT_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel32fC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_FLOAT_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel32fC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_FLOAT_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel32fC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_FLOAT_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel32fC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_FLOAT_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel32fC4A> : std::bool_constant<true>
{
};
#endif

// half float 16
#if (OPPi_ENABLE_HALFFLOAT16_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel16fC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_HALFFLOAT16_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel16fC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_HALFFLOAT16_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel16fC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_HALFFLOAT16_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel16fC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_HALFFLOAT16_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel16fC4A> : std::bool_constant<true>
{
};
#endif

// Bfloat 16
#if (OPPi_ENABLE_BFLOAT16_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel16bfC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_BFLOAT16_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel16bfC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_BFLOAT16_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel16bfC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_BFLOAT16_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel16bfC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_BFLOAT16_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel16bfC4A> : std::bool_constant<true>
{
};
#endif

// int32
#if (OPPi_ENABLE_INT32_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel32sC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT32_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel32sC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT32_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel32sC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT32_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel32sC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT32_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel32sC4A> : std::bool_constant<true>
{
};
#endif

// uint32
#if (OPPi_ENABLE_UINT32_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel32uC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT32_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel32uC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT32_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel32uC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT32_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel32uC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT32_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel32uC4A> : std::bool_constant<true>
{
};
#endif

// int16
#if (OPPi_ENABLE_INT16_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel16sC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT16_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel16sC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT16_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel16sC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT16_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel16sC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT16_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel16sC4A> : std::bool_constant<true>
{
};
#endif

// uint16
#if (OPPi_ENABLE_UINT16_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel16uC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT16_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel16uC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT16_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel16uC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT16_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel16uC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT16_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel16uC4A> : std::bool_constant<true>
{
};
#endif

// int8
#if (OPPi_ENABLE_INT8_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel8sC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT8_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel8sC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT8_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel8sC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT8_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel8sC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_INT8_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel8sC4A> : std::bool_constant<true>
{
};
#endif

// uint8
#if (OPPi_ENABLE_UINT8_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel8uC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT8_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel8uC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT8_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel8uC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT8_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel8uC4> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_UINT8_TYPE) && (OPPi_ENABLE_FOURALPHA_CHANNEL)
template <> struct enable_pixel_type<Pixel8uC4A> : std::bool_constant<true>
{
};
#endif

// complex double
#if (OPPi_ENABLE_COMPLEX_DOUBLE_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel64fcC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_DOUBLE_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel64fcC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_DOUBLE_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel64fcC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_DOUBLE_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel64fcC4> : std::bool_constant<true>
{
};
#endif

// complex float
#if (OPPi_ENABLE_COMPLEX_FLOAT_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel32fcC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_FLOAT_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel32fcC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_FLOAT_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel32fcC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_FLOAT_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel32fcC4> : std::bool_constant<true>
{
};
#endif

// complex half float 16
#if (OPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel16fcC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel16fcC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel16fcC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_HALFFLOAT16_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel16fcC4> : std::bool_constant<true>
{
};
#endif

// complex Bfloat 16
#if (OPPi_ENABLE_COMPLEX_BFLOAT16_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel16bfcC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_BFLOAT16_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel16bfcC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_BFLOAT16_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel16bfcC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_BFLOAT16_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel16bfcC4> : std::bool_constant<true>
{
};
#endif

// complex int32
#if (OPPi_ENABLE_COMPLEX_INT32_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel32scC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_INT32_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel32scC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_INT32_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel32scC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_INT32_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel32scC4> : std::bool_constant<true>
{
};
#endif

// complex int16
#if (OPPi_ENABLE_COMPLEX_INT16_TYPE) && (OPPi_ENABLE_ONE_CHANNEL)
template <> struct enable_pixel_type<Pixel16scC1> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_INT16_TYPE) && (OPPi_ENABLE_TWO_CHANNEL)
template <> struct enable_pixel_type<Pixel16scC2> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_INT16_TYPE) && (OPPi_ENABLE_THREE_CHANNEL)
template <> struct enable_pixel_type<Pixel16scC3> : std::bool_constant<true>
{
};
#endif
#if (OPPi_ENABLE_COMPLEX_INT16_TYPE) && (OPPi_ENABLE_FOUR_CHANNEL)
template <> struct enable_pixel_type<Pixel16scC4> : std::bool_constant<true>
{
};
#endif

} // namespace opp::image