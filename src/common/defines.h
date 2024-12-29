#pragma once
#include <concepts>
#include <cstdint>
#include <type_traits>

// definitions of types, concepts and macros etc. that are used through out in the OPP library. Basically all files
// include this header, so it should't be changed too often...

namespace opp
{
using byte    = std::uint8_t;
using sbyte   = std::int8_t;
using ushort  = std::uint16_t;
using uint    = std::uint32_t;
using ulong64 = std::uint64_t;
using long64  = std::int64_t;

constexpr byte TRUE_VALUE  = 255; // NPP result of comparisons is 255 for "true"
constexpr byte FALSE_VALUE = 0;

struct voidType
{
};

// with these compiler dependent concepts and defines, we can enable/disable
// code segments even if host code never is compiled by nvcc or hipcc
#ifdef __CUDACC__
template <typename T>
concept DEVICE_COMPILER = std::true_type::value;

template <typename T>
concept CUDA_ONLY = std::true_type::value && DEVICE_COMPILER<T>;

template <typename T>
concept HIP_ONLY = std::false_type::value && DEVICE_COMPILER<T>;

template <typename T>
concept HOST_COMPILER = std::false_type::value;

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_CODE __device__ __host__

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_ONLY_CODE __device__

// __restrict__ for device code
#define RESTRICT __restrict__

#define IS_DEVICE_COMPILER
#define IS_CUDA_COMPILER
#elif __HIP_PLATFORM_AMD__
template <typename T>
concept DEVICE_COMPILER = std::true_type::value;

template <typename T>
concept CUDA_ONLY = std::false_type::value;

template <typename T>
concept HIP_ONLY = std::true_type::value;

template <typename T>
concept HOST_COMPILER = std::false_type::value;

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_CODE __device__ __host__

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_ONLY_CODE __device__

// __restrict__ for device code
#define RESTRICT __restrict__

#define IS_DEVICE_COMPILER
#define IS_HIP_COMPILER
#elif _MSC_VER
template <typename T>
concept DEVICE_COMPILER = std::false_type::value;

template <typename T>
concept CUDA_ONLY = std::false_type::value && DEVICE_COMPILER<T>;

template <typename T>
concept HIP_ONLY = std::false_type::value && DEVICE_COMPILER<T>;

template <typename T>
concept HOST_COMPILER = std::true_type::value;

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_CODE

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_ONLY_CODE

// __restrict__ for device code
#define RESTRICT

#define IS_HOST_COMPILER
#elif __clang__
template <typename T>
concept DEVICE_COMPILER = std::false_type::value;

template <typename T>
concept CUDA_ONLY = std::false_type::value && DEVICE_COMPILER<T>;

template <typename T>
concept HIP_ONLY = std::false_type::value && DEVICE_COMPILER<T>;

template <typename T>
concept HOST_COMPILER = std::true_type::value;

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_CODE

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_ONLY_CODE

// __restrict__ for device code
#define RESTRICT

#define IS_HOST_COMPILER
#else // GCC and others
template <typename T>
concept DEVICE_COMPILER = std::false_type::value;

template <typename T>
concept CUDA_ONLY = std::false_type::value && DEVICE_COMPILER<T>;

template <typename T>
concept HIP_ONLY = std::false_type::value && DEVICE_COMPILER<T>;

template <typename T>
concept HOST_COMPILER = std::true_type::value;

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_CODE

// device code needs the __device__ annotation, use the compiler dependent macro:
#define DEVICE_ONLY_CODE

// __restrict__ for device code
#define RESTRICT

#define IS_HOST_COMPILER
#endif

// clang on mac doesn't support parallel execution, so disable it by macro:
#ifdef __APPLE__
#define EXECMODE(type) // NOLINT
#else
#define EXECMODE(type) type, // NOLINT
#endif

template <typename T>
concept HostCode = HOST_COMPILER<T>;

template <typename T>
concept DeviceCode = DEVICE_COMPILER<T>;

template <typename T>
concept HostAndDeviceCode = HOST_COMPILER<T> || DEVICE_COMPILER<T>;

// forward declaration for HalfFp16 and BFloat16
class HalfFp16;
class BFloat16;

// Define our own FP concept as HalfFp16 and BFloat16 are not part of std::floating_point and we don't want to modify
// std namespace
template <typename T>
concept FloatingPoint = std::floating_point<T> || std::same_as<T, HalfFp16> || std::same_as<T, BFloat16>;

// Floating point number of native type, i.e. float or double, but no HalfFp16, BFloat16 etc.
template <typename T>
concept NativeFloatingPoint = std::floating_point<T>;

// Define our own integral concept, who knows, maybe some future new int types... int4?
template <typename T>
concept Integral = std::integral<T>;

// Integer number of native type, i.e. int, short, ushort etc, but not Int4 etc.
template <typename T>
concept NativeIntegral = std::integral<T>;

// all supported native signed integer types
template <typename T>
concept SignedIntegral = std::signed_integral<T>;

// all supported native unsigned integer types
template <typename T>
concept UnsignedIntegral = std::unsigned_integral<T>;

// all supported number formats: floating point and integral types
template <typename T>
concept Number = FloatingPoint<T> || Integral<T>;

// all supported native number formats: floating point and integral types but not fp16, int4 etc
template <typename T>
concept NativeNumber = NativeFloatingPoint<T> || NativeIntegral<T>;

// floating point and signed integral types
template <typename T>
concept SignedNumber = FloatingPoint<T> || SignedIntegral<T>;

// All types that are non native C++ types, currently HalfFp16 and BFloat16
template <typename T>
concept NonNativeType = std::same_as<T, HalfFp16> || std::same_as<T, BFloat16>;

// All types that are native C++ types, all but HalfFp16 and BFloat16
template <typename T>
concept NativeType = !NonNativeType<T>;

// Is of type BFloat16
template <typename T>
concept IsBFloat16 = std::same_as<T, BFloat16>;

// Is of type HalfFp16
template <typename T>
concept IsHalfFp16 = std::same_as<T, HalfFp16>;

// T is sizeof 1 byte
template <typename T>
concept ByteSizeType = sizeof(T) == 1;

// T is sizeof 2 bytes
template <typename T>
concept TwoBytesSizeType = sizeof(T) == 2;

// T is sizeof 4 bytes
template <typename T>
concept FourBytesSizeType = sizeof(T) == 4;

// size is Power of 2
template <std::size_t size>
concept IsPowerOf2 = (size != 0) && ((size & (size - 1)) == 0);

// Set to true to enable SIMD versions of implementation (applies only when both implementations exist)
template <typename T>
concept EnableSIMD = std::true_type::value;

// GCC complains if we directly use std::false_type in a static_assert
// (we do that from time to time for checking if we missed a use case)
template <typename T> struct AlwaysFalse : std::false_type
{
};

template <typename T> struct AlwaysTrue : std::true_type
{
};

} // namespace opp