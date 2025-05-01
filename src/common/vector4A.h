#pragma once
#include "defines.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "opp_defs.h"
#include "vector4.h"
#include "vector_typetraits.h"
#include <common/utilities.h>
#include <concepts>
#include <iostream>

namespace opp
{

// forward declaration:
template <Number T> struct Vector1;
template <Number T> struct Vector2;
template <Number T> struct Vector4;

/// <summary>
/// A four T component vector. Operations are performed on the first three channels, W is treated as additional Alpha
/// channel and remains unused. Can replace CUDA's vector4 types
/// </summary>
template <Number T> struct alignas(4 * sizeof(T)) Vector4A
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
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec, RoundingMode aRoundingMode) noexcept
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
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec, RoundingMode aRoundingMode) noexcept
        requires IsHalfFp16<T> && IsFloat<T2>;

    /// <summary>
    /// Type conversion for complex with rounding (only for float to bfloat/halffloat)
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector4A(const Vector4A<T2> &aVec, RoundingMode aRoundingMode) noexcept
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
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector4A<T> &AbsDiff(const Vector4A<T> &aOther)
        requires RealSignedNumber<T> && NonNativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires RealSignedNumber<T> && NonNativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector4A<T> &AbsDiff(const Vector4A<T> &aOther)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

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
        requires DeviceCode<T> && NativeFloatingPoint<T>;

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

template <HostCode T2> std::ostream &operator<<(std::ostream &aOs, const Vector4A<T2> &aVec);

template <HostCode T2> std::wostream &operator<<(std::wostream &aOs, const Vector4A<T2> &aVec);

// byte and sbyte are treated as characters and not numbers:
template <HostCode T2>
std::ostream &operator<<(std::ostream &aOs, const Vector4A<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
std::wostream &operator<<(std::wostream &aOs, const Vector4A<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2> std::istream &operator>>(std::istream &aIs, Vector4A<T2> &aVec);

template <HostCode T2> std::wistream &operator>>(std::wistream &aIs, Vector4A<T2> &aVec);

template <HostCode T2>
std::istream &operator>>(std::istream &aIs, Vector4A<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
std::wistream &operator>>(std::wistream &aIs, Vector4A<T2> &aVec)
    requires ByteSizeType<T2>;

} // namespace opp