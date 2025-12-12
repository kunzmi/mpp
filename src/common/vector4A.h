#pragma once
#include "defines.h"
#include "dllexport_common.h"
#include "mpp_defs.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "vector_typetraits.h"
#include "vector4.h"
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/half_fp16.h>
#include <common/utilities.h>
#include <concepts>
#include <iostream>

namespace mpp
{

// forward declaration:
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector1;
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector2;
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector4;

/// <summary>
/// A four T component vector. Operations are performed on the first three channels, W is treated as additional Alpha
/// channel and remains unused. Can replace CUDA's vector4 types
/// </summary>
template <Number T> struct alignas(4 * sizeof(T)) MPPEXPORT_COMMON Vector4A
{
    T x; // NOLINT
    T y; // NOLINT
    T z; // NOLINT
    T w; // NOLINT

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    Vector4A() noexcept = default; // NOLINT

    /// <summary>
    /// Initializes vector to all components = aVal, except w
    /// </summary>
    DEVICE_CODE constexpr Vector4A(T aVal) noexcept : x(aVal), y(aVal), z(aVal) // NOLINT
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal, except w (especially when set to 0)
    /// </summary>
    DEVICE_CODE constexpr Vector4A(int aVal) noexcept // NOLINT
        requires(!IsInt<T>)
        : x(static_cast<T>(aVal)), y(static_cast<T>(aVal)), z(static_cast<T>(aVal))
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0], aVal[1], aVal[2]], w remains unitialized
    /// </summary>
    DEVICE_CODE constexpr explicit Vector4A(T aVal[3]) noexcept // NOLINT
        : x(aVal[0]), y(aVal[1]), z(aVal[2])
    {
    }

    /// <summary>
    /// Initializes vector to [aX, aY, aZ], w remains unitialized
    /// </summary>
    DEVICE_CODE constexpr Vector4A(T aX, T aY, T aZ) noexcept : x(aX), y(aY), z(aZ) // NOLINT
    {
    }

    /// <summary>
    /// Usefull constructor if we want a Vector4A from 3 channel pixel Vector3, w remains unitialized
    /// </summary>
    DEVICE_CODE constexpr Vector4A(const Vector3<T> &aVec3) noexcept : x(aVec3.x), y(aVec3.y), z(aVec3.z) // NOLINT
    {
    }

    ///// <summary>
    ///// Usefull constructor if we want a Vector4A from 4 channel pixel Vector4
    ///// </summary>
    // DEVICE_CODE constexpr Vector4A(const Vector4<T> &aVec4) noexcept
    //     : x(aVec4.x), y(aVec4.y), z(aVec4.z) // NOLINT
    // {
    // }

    /// <summary>
    /// Usefull constructor if we want a Vector4A from 4 channel pixel Vector4
    /// </summary>
    DEVICE_CODE constexpr Vector4A( // NOLINT
        const Vector4<T> &aVec4) noexcept
        // requires ByteSizeType<T> || TwoBytesSizeType<T>
        : x(aVec4.x), y(aVec4.y), z(aVec4.z), w(aVec4.w)
    {
        // In case of one or two byte base types, it is probably more efficient to set 32 or 64 bits in one go rather
        // than split it up in smaller words. Thus also initialize w.

        // GCC 13 has a bug with explicit template instantiation and concepts: we get internal compiler errors if we
        // activate both variants, so just init all 4 members for all types and switch to GCC 14 one day...
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector4A FromUint(const uint &aUint) noexcept
        requires ByteSizeType<T>;

    /// <summary>
    /// Usefull constructor for SIMD instructions (performs the bitshifts needed to merge compare results for 2 byte
    /// types)
    /// </summary>
    DEVICE_CODE static Vector4A FromUint(uint aUintLO, uint aUintHI) noexcept
        requires IsByte<T>;

    /// <summary>
    /// Type conversion with saturation if needed, w remains unitialized<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <Number T2> DEVICE_CODE Vector4A(const Vector4A<T2> &aVec) noexcept; // NOLINT

    /// <summary>
    /// Type conversion with saturation if needed, w remains unitialized<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <Number T2>
    DEVICE_CODE explicit Vector4A( // NOLINT
        Vector4A<T2> &aVec) noexcept
        // Disable the non-const variant for half and bfloat to / from float,
        // otherwise the const specialization will never be picked up:
        requires(!(IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>) &&
                 !(IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>)) &&
                (!std::same_as<T, T2>);

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to BFloat2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec) noexcept // NOLINT
        requires IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for BFloat2 to float2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec) noexcept // NOLINT
        requires IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to half2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec) noexcept // NOLINT
        requires IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to BFloat2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec, RoundingMode aRoundingMode)
        requires IsBFloat16<T> && IsFloat<T2>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for half2 to float2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec) noexcept // NOLINT
        requires IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to half2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec, RoundingMode aRoundingMode)
        requires IsHalfFp16<T> && IsFloat<T2>;

    /// <summary>
    /// Type conversion for complex with rounding (only for float to bfloat/halffloat)
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec, RoundingMode aRoundingMode)
        requires ComplexFloatingPoint<T> && ComplexFloatingPoint<T2> &&
                 NonNativeFloatingPoint<complex_basetype_t<remove_vector_t<T>>> &&
                 std::same_as<float, complex_basetype_t<remove_vector_t<T2>>>;

    ~Vector4A() = default;

    Vector4A(const Vector4A &) noexcept            = default;
    Vector4A(Vector4A &&) noexcept                 = default;
    Vector4A &operator=(const Vector4A &) noexcept = default;
    Vector4A &operator=(Vector4A &&) noexcept      = default;

  private:
    // if we make those converter public we will get in trouble with some T constructors / operators
    /// <summary>
    /// converter to uint for SIMD operations
    /// </summary>
    DEVICE_CODE operator const uint &() const // NOLINT
        requires ByteSizeType<T>;

    /// <summary>
    /// converter to uint for SIMD operations
    /// </summary>
    DEVICE_CODE operator uint &() // NOLINT
        requires ByteSizeType<T>;

#pragma endregion
  public:
#pragma region Operators
    // don't use space-ship operator as it returns true if any comparison returns true.
    // But NPP only returns true if all channels fulfill the comparison.
    // auto operator<=>(const Vector4A &) const = default;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector4A &aLeft, const Vector4A &aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector4A &aLeft, const Vector4A &aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector4A &aLeft, const Vector4A &aRight, T aEpsilon)
        requires Is16BitFloat<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector4A &aLeft, const Vector4A &aRight,
                                                complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Returns true if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(const Vector4A &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(const Vector4A &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(const Vector4A &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(const Vector4A &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(const Vector4A &aOther) const;

    /// <summary>
    /// Returns true if any element comparison is true, ignoring alpha / w-value
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(const Vector4A &aOther) const;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector4A &aOther) const
        requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector4A &aOther) const
        requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector4A &aOther) const
        requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector4A &aOther) const
        requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector4A &aOther) const
        requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector4A &aOther) const
        requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector4A &aOther) const
        requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector4A &aOther) const
        requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector4A &aOther) const
        requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector4A &aOther) const
        requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector4A &aOther) const
        requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector4A &aOther) const
        requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector4A &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector4A &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector4A &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector4A &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector4A &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector4A &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector4A &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector4A &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector4A &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector4A &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector4A &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector4A &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator-() const
        requires RealSignedNumber<T> || ComplexNumber<T>;

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-() const
        requires IsSByte<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-() const
        requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-() const
        requires IsBFloat16<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-() const
        requires IsHalfFp16<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector4A &operator+=(T aOther);

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector4A &operator+=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector4A &operator+=(const Vector4A &aOther);

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator+=(const Vector4A &aOther)
        requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator+=(const Vector4A &aOther)
        requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator+=(const Vector4A &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator+=(const Vector4A &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator+=(const Vector4A &aOther)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator+=(const Vector4A &aOther)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator+(const Vector4A &aOther) const;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator+(const Vector4A &aOther) const
        requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator+(const Vector4A &aOther) const
        requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator+(const Vector4A &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator+(const Vector4A &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator+(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator+(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector4A &operator-=(T aOther);

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector4A &operator-=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector4A &operator-=(const Vector4A &aOther);

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator-=(const Vector4A &aOther)
        requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator-=(const Vector4A &aOther)
        requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator-=(const Vector4A &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator-=(const Vector4A &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator-=(const Vector4A &aOther)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator-=(const Vector4A &aOther)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector4A &SubInv(const Vector4A &aOther);

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &SubInv(const Vector4A &aOther)
        requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &SubInv(const Vector4A &aOther)
        requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &SubInv(const Vector4A &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &SubInv(const Vector4A &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &SubInv(const Vector4A &aOther)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &SubInv(const Vector4A &aOther)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator-(const Vector4A &aOther) const;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-(const Vector4A &aOther) const
        requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-(const Vector4A &aOther) const
        requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-(const Vector4A &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-(const Vector4A &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator-(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector4A &operator*=(T aOther);

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector4A &operator*=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector4A &operator*=(const Vector4A &aOther);

    /// <summary>
    /// Component wise multiplication SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator*=(const Vector4A &aOther)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise multiplication SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator*=(const Vector4A &aOther)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator*(const Vector4A &aOther) const;

    /// <summary>
    /// Component wise multiplication SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator*(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise multiplication SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator*(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector4A &operator/=(T aOther);

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector4A &operator/=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector4A &operator/=(const Vector4A &aOther);

    /// <summary>
    /// Component wise division SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator/=(const Vector4A &aOther)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise division SIMD
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &operator/=(const Vector4A &aOther)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise division (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInv(const Vector4A &aOther);

    /// <summary>
    /// Component wise division SIMD (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &DivInv(const Vector4A &aOther)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise division SIMD (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A &DivInv(const Vector4A &aOther)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator/(const Vector4A &aOther) const;

    /// <summary>
    /// Inplace Integer division with element wise round()
    /// </summary>
    DEVICE_CODE Vector4A &DivRound(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector4A &DivRoundNearest(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector4A &DivRoundZero(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE Vector4A &DivFloor(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE Vector4A &DivCeil(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvRound(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even (inverted inplace div: this =
    /// aOther / this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvRoundNearest(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero (inverted inplace div: this = aOther /
    /// this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvRoundZero(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvFloor(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvCeil(const Vector4A &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivRound(const Vector4A &aLeft, const Vector4A &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivRoundNearest(const Vector4A &aLeft, const Vector4A &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivRoundZero(const Vector4A &aLeft, const Vector4A &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivFloor(const Vector4A &aLeft, const Vector4A &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivCeil(const Vector4A &aLeft, const Vector4A &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleRound(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round nearest ties to even (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleRoundNearest(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round toward zero (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleRoundZero(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise floor (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleFloor(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise ceil() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleCeil(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round()
    /// </summary>
    DEVICE_CODE Vector4A &DivRound(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector4A &DivRoundNearest(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector4A &DivRoundZero(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE Vector4A &DivFloor(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE Vector4A &DivCeil(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvRound(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even (inverted inplace div: this =
    /// aOther / this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvRoundNearest(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero (inverted inplace div: this = aOther /
    /// this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvRoundZero(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvFloor(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector4A &DivInvCeil(const Vector4A &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivRound(const Vector4A &aLeft, const Vector4A &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivRoundNearest(const Vector4A &aLeft, const Vector4A &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivRoundZero(const Vector4A &aLeft, const Vector4A &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivFloor(const Vector4A &aLeft, const Vector4A &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A DivCeil(const Vector4A &aLeft, const Vector4A &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleRound(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round nearest ties to even (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleRoundNearest(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round toward zero (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleRoundZero(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise floor (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleFloor(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise ceil() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector4A &DivScaleCeil(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Component wise division SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator/(const Vector4A &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise division SIMD
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] Vector4A operator/(const Vector4A &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T &operator[](Axis4D aAxis) const
        requires DeviceCode<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] const T &operator[](Axis4D aAxis) const
        requires HostCode<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] T &operator[](Axis4D aAxis)
        requires DeviceCode<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] T &operator[](Axis4D aAxis)
        requires HostCode<T>;
#pragma endregion

#pragma region Convert Methods
    /// <summary>
    /// Type conversion without saturation, direct type conversion
    /// </summary>
    template <Number T2> [[nodiscard]] static Vector4A<T> DEVICE_CODE Convert(const Vector4A<T2> &aVec);
#pragma endregion

#pragma region Integral only Methods
    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector4A<T> &LShift(const Vector4A<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> LShift(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector4A<T> &RShift(const Vector4A<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> RShift(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector4A<T> &LShift(uint aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> LShift(const Vector4A<T> &aLeft, uint aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector4A<T> &RShift(uint aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> RShift(const Vector4A<T> &aLeft, uint aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE Vector4A<T> &And(const Vector4A<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> And(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE Vector4A<T> &Or(const Vector4A<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Or(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE Vector4A<T> &Xor(const Vector4A<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Xor(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE Vector4A<T> &Not()
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Not(const Vector4A<T> &aVec)
        requires RealIntegral<T>;
#pragma endregion

#pragma region Methods
#pragma region Exp
    /// <summary>
    /// Element wise exponential
    /// </summary>
    Vector4A<T> &Exp()
        requires HostCode<T> && NativeNumber<T>;
    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector4A<T> &Exp()
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T>) || ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector4A<T> &Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise exponential (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Exp()
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise exponential (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Exp()
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Exp(const Vector4A<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Exp(const Vector4A<T> &aVec)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Exp(const Vector4A<T> &aVec)
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T>) || ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise exponential (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Exp(const Vector4A<T> &aVec)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise exponential (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Exp(const Vector4A<T> &aVec)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;
#pragma endregion

#pragma region Log
    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    Vector4A<T> &Ln()
        requires HostCode<T> && NativeNumber<T>;
    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector4A<T> &Ln()
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector4A<T> &Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Ln()
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise natural logarithm (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Ln()
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Ln(const Vector4A<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Ln(const Vector4A<T> &aVec)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Ln(const Vector4A<T> &aVec)
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Ln(const Vector4A<T> &aVec)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise natural logarithm (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Ln(const Vector4A<T> &aVec)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;
#pragma endregion

#pragma region Sqr
    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE Vector4A<T> &Sqr();

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Sqr(const Vector4A<T> &aVec);
#pragma endregion

#pragma region Sqrt
    /// <summary>
    /// Element wise square root
    /// </summary>
    Vector4A<T> &Sqrt()
        requires HostCode<T> && NativeNumber<T>;
    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector4A<T> &Sqrt()
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector4A<T> &Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise square root (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Sqrt()
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise square root (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Sqrt()
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Sqrt(const Vector4A<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Sqrt(const Vector4A<T> &aVec)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Sqrt(const Vector4A<T> &aVec)
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise square root (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Sqrt(const Vector4A<T> &aVec)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise square root (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Sqrt(const Vector4A<T> &aVec)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;
#pragma endregion

#pragma region Abs
    /// <summary>
    /// Element wise absolute
    /// </summary>
    Vector4A<T> &Abs()
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector4A<T> &Abs()
        requires RealSignedNumber<T> && NonNativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector4A<T> &Abs()
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Abs()
        requires IsSByte<T> && CUDA_ONLY<T> && RealSignedNumber<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Abs()
        requires IsShort<T> && CUDA_ONLY<T> && RealSignedNumber<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Abs()
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Abs()
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Abs(const Vector4A<T> &aVec)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Abs(const Vector4A<T> &aVec)
        requires IsSByte<T> && CUDA_ONLY<T> && RealSignedNumber<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Abs(const Vector4A<T> &aVec)
        requires IsShort<T> && CUDA_ONLY<T> && RealSignedNumber<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Abs(const Vector4A<T> &aVec)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Abs(const Vector4A<T> &aVec)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Abs(const Vector4A<T> &aVec)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Abs(const Vector4A<T> &aVec)
        requires RealSignedNumber<T> && NonNativeNumber<T>;
#pragma endregion

#pragma region AbsDiff
    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector4A<T> &AbsDiff(const Vector4A<T> &aOther)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector4A<T> &AbsDiff(const Vector4A<T> &aOther)
        requires RealSignedNumber<T> && NonNativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealSignedNumber<T> && NonNativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector4A<T> &AbsDiff(const Vector4A<T> &aOther)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &AbsDiff(const Vector4A<T> &aOther)
        requires IsSByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &AbsDiff(const Vector4A<T> &aOther)
        requires IsByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &AbsDiff(const Vector4A<T> &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &AbsDiff(const Vector4A<T> &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsSByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;
#pragma endregion

#pragma region Methods for Complex types
    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE Vector4A<T> &Conj()
        requires ComplexNumber<T>;

    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Conj(const Vector4A<T> &aValue)
        requires ComplexNumber<T>;

    /// <summary>
    /// Conjugate complex multiplication: this * conj(aOther)  per element
    /// </summary>
    DEVICE_CODE Vector4A<T> &ConjMul(const Vector4A<T> &aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Conjugate complex multiplication: aLeft * conj(aRight) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> ConjMul(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires ComplexNumber<T>;

    /// <summary>
    /// Complex magnitude per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A<complex_basetype_t<T>> Magnitude() const
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Complex magnitude squared per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A<complex_basetype_t<T>> MagnitudeSqr() const
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Angle between real and imaginary of a complex number (atan2(image, real)) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A<complex_basetype_t<T>> Angle() const
        requires ComplexFloatingPoint<T>;
#pragma endregion

#pragma region Clamp
    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector4A<T> &Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    Vector4A<T> &Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector4A<T> &Clamp(T aMinVal, T aMaxVal)
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector4A<T> &Clamp(complex_basetype_t<T> aMinVal, complex_basetype_t<T> aMaxVal)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector4A<T> &ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>);

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector4A<T> &ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>);
#pragma endregion

#pragma region Min
    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector4A<T> &Min(const Vector4A<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Min(const Vector4A<T> &aOther)
        requires IsSByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Min(const Vector4A<T> &aOther)
        requires IsByte<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Min(const Vector4A<T> &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Min(const Vector4A<T> &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Min(const Vector4A<T> &aOther)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Min(const Vector4A<T> &aOther)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    Vector4A<T> &Min(const Vector4A<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector4A<T> &Min(const Vector4A<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>));

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsSByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsBFloat16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsHalfFp16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>));

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    [[nodiscard]] T Min() const
        requires HostCode<T> && NativeNumber<T>;
#pragma endregion

#pragma region Max
    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector4A<T> &Max(const Vector4A<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Max(const Vector4A<T> &aOther)
        requires IsSByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Max(const Vector4A<T> &aOther)
        requires IsByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Max(const Vector4A<T> &aOther)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Max(const Vector4A<T> &aOther)
        requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Max(const Vector4A<T> &aOther)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Max(const Vector4A<T> &aOther)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    Vector4A<T> &Max(const Vector4A<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector4A<T> &Max(const Vector4A<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>));

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsSByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsByte<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>));

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    [[nodiscard]] T Max() const
        requires HostCode<T> && NativeNumber<T>;
#pragma endregion

#pragma region Round
    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Round(const Vector4A<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Vector4A<T> &Round()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Vector4A<T> &Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    Vector4A<T> &Round()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Floor(const Vector4A<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE Vector4A<T> &Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    Vector4A<T> &Floor()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Floor()
        requires IsBFloat16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise floor() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Floor()
        requires IsHalfFp16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE Vector4A<T> &Floor()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Ceil(const Vector4A<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE Vector4A<T> &Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    Vector4A<T> &Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Ceil()
        requires IsBFloat16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise ceil() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &Ceil()
        requires IsHalfFp16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE Vector4A<T> &Ceil()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> RoundNearest(const Vector4A<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector4A<T> &RoundNearest()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even <para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    Vector4A<T> &RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &RoundNearest()
        requires IsBFloat16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise round nearest ties to even (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &RoundNearest()
        requires IsHalfFp16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector4A<T> &RoundNearest()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> RoundZero(const Vector4A<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector4A<T> &RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    Vector4A<T> &RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &RoundZero()
        requires IsBFloat16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise round toward zero (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector4A<T> &RoundZero()
        requires IsHalfFp16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector4A<T> &RoundZero()
        requires NonNativeFloatingPoint<T>;
#pragma endregion

#pragma region Compare per element
    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    [[nodiscard]] static Vector4A<byte> CompareEQEps(const Vector4A<T> &aLeft, const Vector4A<T> &aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<byte> CompareEQEps(const Vector4A<T> &aLeft, const Vector4A<T> &aRight,
                                                                 T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<byte> CompareEQEps(const Vector4A<T> &aLeft, const Vector4A<T> &aRight,
                                                                 T aEpsilon)
        requires Is16BitFloat<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<byte> CompareEQEps(const Vector4A<T> &aLeft, const Vector4A<T> &aRight,
                                                                 complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>;
    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<byte> CompareEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight);

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<byte> CompareGE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<byte> CompareGT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<byte> CompareLE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<byte> CompareLT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<byte> CompareNEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight);

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareNEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsSByte<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareNEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsSByte<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareNEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareNEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareNEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareGT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLE(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareLT(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector4A<byte> CompareNEQ(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;
#pragma endregion

#pragma region Data accessors
    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<T> XYZ() const;

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<T> YZW() const;

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> XY() const;

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> YZ() const;

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> ZW() const;

    /// <summary>
    /// Provide a smiliar accessor to inner data as for std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] T *data();

    /// <summary>
    /// Provide a smiliar accessor to inner data as for std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T *data() const;
#pragma endregion
#pragma endregion
};

template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator+(const Vector4A<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector4A<T>{static_cast<T>(aLeft.x + aRight), static_cast<T>(aLeft.y + aRight),
                       static_cast<T>(aLeft.z + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator+(T2 aLeft, const Vector4A<T> &aRight)
    requires Number<T2>
{
    return Vector4A<T>{static_cast<T>(aLeft + aRight.x), static_cast<T>(aLeft + aRight.y),
                       static_cast<T>(aLeft + aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator-(const Vector4A<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector4A<T>{static_cast<T>(aLeft.x - aRight), static_cast<T>(aLeft.y - aRight),
                       static_cast<T>(aLeft.z - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator-(T2 aLeft, const Vector4A<T> &aRight)
    requires Number<T2>
{
    return Vector4A<T>{static_cast<T>(aLeft - aRight.x), static_cast<T>(aLeft - aRight.y),
                       static_cast<T>(aLeft - aRight.z)};
}

template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator*(const Vector4A<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector4A<T>{static_cast<T>(aLeft.x * aRight), static_cast<T>(aLeft.y * aRight),
                       static_cast<T>(aLeft.z * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator*(T2 aLeft, const Vector4A<T> &aRight)
    requires Number<T2>
{
    return Vector4A<T>{static_cast<T>(aLeft * aRight.x), static_cast<T>(aLeft * aRight.y),
                       static_cast<T>(aLeft * aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator/(const Vector4A<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector4A<T>{static_cast<T>(aLeft.x / aRight), static_cast<T>(aLeft.y / aRight),
                       static_cast<T>(aLeft.z / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator/(T2 aLeft, const Vector4A<T> &aRight)
    requires Number<T2>
{
    return Vector4A<T>{static_cast<T>(aLeft / aRight.x), static_cast<T>(aLeft / aRight.y),
                       static_cast<T>(aLeft / aRight.z)};
}

template <HostCode T2> MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<T2> &aVec);

template <HostCode T2> MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<T2> &aVec);

// byte and sbyte are treated as characters and not numbers:
template <HostCode T2>
MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2> MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<T2> &aVec);

template <HostCode T2> MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<T2> &aVec);

template <HostCode T2>
MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<T2> &aVec)
    requires ByteSizeType<T2>;

#ifdef IS_HOST_COMPILER
extern template struct Vector4A<sbyte>;
extern template struct Vector4A<byte>;
extern template struct Vector4A<short>;
extern template struct Vector4A<ushort>;
extern template struct Vector4A<int>;
extern template struct Vector4A<uint>;
extern template struct Vector4A<long64>;
extern template struct Vector4A<ulong64>;

extern template struct Vector4A<BFloat16>;
extern template struct Vector4A<HalfFp16>;
extern template struct Vector4A<float>;
extern template struct Vector4A<double>;

extern template struct Vector4A<Complex<sbyte>>;
extern template struct Vector4A<Complex<short>>;
extern template struct Vector4A<Complex<int>>;
extern template struct Vector4A<Complex<long64>>;
extern template struct Vector4A<Complex<BFloat16>>;
extern template struct Vector4A<Complex<HalfFp16>>;
extern template struct Vector4A<Complex<float>>;
extern template struct Vector4A<Complex<double>>;

extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<double> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(const Vector4A<float> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<double> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(const Vector4A<float> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(const Vector4A<float> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<sbyte> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(const Vector4A<short> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(const Vector4A<int> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(const Vector4A<long64> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<float>> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<float>> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(const Vector4A<float> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(const Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<short>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<int>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<float>::Vector4A(Vector4A<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<double>::Vector4A(Vector4A<float> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<float>> &) noexcept;

extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<byte> &Vector4A<byte>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<short> &Vector4A<short>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<int> &Vector4A<int>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<uint> &Vector4A<uint>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<long64> &Vector4A<long64>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<float> &Vector4A<float>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<double> &Vector4A<double>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    byte>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    int>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    uint>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<sbyte> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<sbyte> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<sbyte> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<sbyte> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<byte> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<byte> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<byte> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<byte> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<short> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<short> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<short> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<short> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<ushort> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<ushort> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<ushort> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<ushort> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<int> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<int> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<int> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<int> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<uint> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<uint> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<uint> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<uint> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<long64> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<long64> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<long64> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<long64> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<ulong64> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<ulong64> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<ulong64> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<ulong64> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<BFloat16> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<HalfFp16> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<float> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<float> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<float> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<float> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<double> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<double> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<double> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<double> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<sbyte>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<sbyte>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<Complex<sbyte>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<sbyte>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<short>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<short>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<Complex<short>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<short>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<int>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<int>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<Complex<int>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<int>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<long64>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<long64>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<Complex<long64>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<long64>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<BFloat16>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<BFloat16>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<Complex<BFloat16>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<BFloat16>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<HalfFp16>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<HalfFp16>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<Complex<HalfFp16>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<HalfFp16>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<float>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<float>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<Complex<float>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<float>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<double>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<double>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector4A<Complex<double>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<double>> &aVec);
#endif // IS_HOST_COMPILER
} // namespace mpp