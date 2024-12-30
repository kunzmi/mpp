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