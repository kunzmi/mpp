#pragma once
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace opp
{
using byte    = std::uint8_t;
using sbyte   = std::int8_t;
using ushort  = std::uint16_t;
using uint    = std::uint32_t;
using ulong64 = std::uint64_t;
using long64  = std::int64_t;

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
#define PRAGMA_UNROLL _Pragma("unroll")
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
#define PRAGMA_UNROLL _Pragma("unroll")
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
// pragma unroll is only for device code
#define PRAGMA_UNROLL
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
// pragma unroll is only for device code
#define PRAGMA_UNROLL
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
// pragma unroll is only for device code
#define PRAGMA_UNROLL
#endif

template <typename T>
concept HostCode = HOST_COMPILER<T>;

template <typename T>
concept DeviceCode = DEVICE_COMPILER<T>;

template <typename T>
concept HostAndDeviceCode = HOST_COMPILER<T> || DEVICE_COMPILER<T>;

// Define our own FP concept as we might add FP16, Bfloat16 and FP8 one day...
template <typename T>
concept FloatingPoint = std::floating_point<T>;

// Floating point number of native type, i.e. float or double, but no FP16, Bfloat16 etc.
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

// T is sizeof 1 byte
template <typename T>
concept ByteSizeType = sizeof(T) == 1;

// size is Power of 2
template <std::size_t size>
concept IsPowerOf2 = (size != 0) && ((size & (size - 1)) == 0);

// size is Power of 2 and <= 16
template <std::size_t size>
concept IsTupelSize = IsPowerOf2<size> && size <= 16;

// GCC complains if we directly use std::false_type in a static_assert
// (we do that from time to time for checking if we missed a use case)
template <typename T> struct AlwaysFalse : std::false_type
{
};

template <typename T> struct AlwaysTrue : std::true_type
{
};

// template <size_t N> struct KernelNameWrapper
//{
//     constexpr KernelNameWrapper(const char (&kernelName)[N])
//     {
//         for (size_t i = 0; i < N; i++)
//         {
//             value[i] = kernelName[i];
//         }
//     }
//
//     char value[N];
// };
//
// template <KernelNameWrapper kernelName, typename T, int ComputeCapability = -1> struct blockSizeConfig
//{
//     static constexpr uint value[] = {32, 8, 1};
// };
// template <int ComputeCapability, typename T> struct blockSizeConfig<"hallo", T, ComputeCapability>
//{
//     static constexpr uint value[] = {16, 16, 1};
// };
// template <int ComputeCapability, typename T> struct blockSizeConfig<"hallo2", T, ComputeCapability>
//{
//     static constexpr uint value[] = {8, 8, 1};
// };
//
// template <KernelNameWrapper kernelName, int ComputeCapability = -1, typename T = void> struct bytesPerWarpConfig
//{
//     static constexpr int value = 256;
// };
//
// template <KernelNameWrapper kernelName, int ComputeCapability = -1, typename T = void> struct
// dynamicSharedMemoryConfig
//{
//     static constexpr int value = 0;
// };
//
// template <KernelNameWrapper kernelName, int ComputeCapability = -1, typename T = void> struct kernelLaunchConfig
//{
//     static constexpr uint value[]            = blockSizeConfig<kernelName, ComputeCapability, T>::value;
//     static constexpr int bytesPerWarp        = bytesPerWarpConfig<kernelName, ComputeCapability, T>::value;
//     static constexpr int dynamicSharedMemory = dynamicSharedMemoryConfig<kernelName, ComputeCapability, T>::value;
// };

/// <summary>
/// Rounding Modes<para/>
/// The enumerated rounding modes are used by a large number of OPP primitives
/// to allow the user to specify the method by which fractional values are converted
/// to integer values.
/// </summary>
enum class RoudingMode
{
    /// <summary>
    /// Round to the nearest even integer.<para/>
    /// All fractional numbers are rounded to their nearest integer. The ambiguous
    /// cases (i.e. integer.5) are rounded to the closest even integer.<para/>
    /// __float2int_rn in CUDA<para/>
    /// E.g.<para/>
    /// - roundNear(0.4) = 0<para/>
    /// - roundNear(0.5) = 0<para/>
    /// - roundNear(0.6) = 1<para/>
    /// - roundNear(1.5) = 2<para/>
    /// - roundNear(1.9) = 2<para/>
    /// - roundNear(-1.5) = -2<para/>
    /// - roundNear(-2.5) = -2<para/>
    /// </summary>
    NearestTiesToEven,
    /// <summary>
    /// Round according to financial rule.<para/>
    /// All fractional numbers are rounded to their nearest integer. The ambiguous
    /// cases (i.e. integer.5) are rounded away from zero.<para/>
    /// C++ round() function<para/>
    /// E.g.<para/>
    /// - roundNearestTiesAwayFromZero(0.4)  = 0<para/>
    /// - roundNearestTiesAwayFromZero(0.5)  = 1<para/>
    /// - roundNearestTiesAwayFromZero(0.6)  = 1<para/>
    /// - roundNearestTiesAwayFromZero(1.5)  = 2<para/>
    /// - roundNearestTiesAwayFromZero(1.9)  = 2<para/>
    /// - roundNearestTiesAwayFromZero(-1.5) = -2<para/>
    /// - roundNearestTiesAwayFromZero(-2.5) = -3<para/>
    /// </summary>
    NearestTiesAwayFromZero,
    /// <summary>
    /// Round towards zero (truncation).<para/>
    /// All fractional numbers of the form integer.decimals are truncated to
    /// __float2int_rz in CUDA<para/>
    /// integer.<para/>
    /// - roundZero(0.4)  = 0<para/>
    /// - roundZero(0.5)  = 0<para/>
    /// - roundZero(0.6)  = 0<para/>
    /// - roundZero(1.5)  = 1<para/>
    /// - roundZero(1.9)  = 1<para/>
    /// - roundZero(-1.5) = -1<para/>
    /// - roundZero(-2.5) = -2<para/>
    /// </summary>
    TowardZero,
    /// <summary>
    /// Round towards negative infinity.<para/>
    /// C++ floor() function<para/>
    /// E.g.<para/>
    /// - floor(0.4)  = 0<para/>
    /// - floor(0.5)  = 0<para/>
    /// - floor(0.6)  = 0<para/>
    /// - floor(1.5)  = 1<para/>
    /// - floor(1.9)  = 1<para/>
    /// - floor(-1.5) = -2<para/>
    /// - floor(-2.5) = -3<para/>
    /// </summary>
    TowardNegativeInfinity,
    /// <summary>
    /// Round towards positive infinity.<para/>
    /// C++ ceil() function<para/>
    /// E.g.<para/>
    /// - ceil(0.4)  = 1<para/>
    /// - ceil(0.5)  = 1<para/>
    /// - ceil(0.6)  = 1<para/>
    /// - ceil(1.5)  = 2<para/>
    /// - ceil(1.9)  = 2<para/>
    /// - ceil(-1.5) = -1<para/>
    /// - ceil(-2.5) = -2<para/>
    /// </summary>
    TowardPositiveInfinity
};

} // namespace opp