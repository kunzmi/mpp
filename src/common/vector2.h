#pragma once
#include "defines.h"
#include "dllexport_common.h"
#include "mpp_defs.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "vector_typetraits.h"
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/half_fp16.h>
#include <concepts>
#include <iostream>
#include <type_traits>

namespace mpp
{

// forward declaration:
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector1;
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector2;
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector3;
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector4;

enum class Axis2D // NOLINT(performance-enum-size)
{
    X = 0,
    Y = 1
};

MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Axis2D &aAxis);
MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Axis2D &aAxis);

/// <summary>
/// A two T component vector. Can replace CUDA's vector2 types
/// </summary>
template <Number T> struct alignas(2 * sizeof(T)) MPPEXPORT_COMMON Vector2
{
    T x; // NOLINT
    T y; // NOLINT

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    Vector2() noexcept = default;

    /// <summary>
    /// Initializes vector to all components = aVal
    /// </summary>
    DEVICE_CODE constexpr Vector2(T aVal) noexcept : x(aVal), y(aVal) // NOLINT
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal (especially when set to 0)
    /// </summary>
    DEVICE_CODE constexpr Vector2(int aVal) noexcept // NOLINT
        requires(!IsInt<T>)
        : x(static_cast<T>(aVal)), y(static_cast<T>(aVal))
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0], aVal[1]]
    /// </summary>
    DEVICE_CODE constexpr explicit Vector2(T aVal[2]) noexcept // NOLINT
        : x(aVal[0]), y(aVal[1])
    {
    }

    /// <summary>
    /// Initializes vector to [aX, aY]
    /// </summary>
    DEVICE_CODE constexpr Vector2(T aX, T aY) noexcept : x(aX), y(aY) // NOLINT
    {
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector2 FromUint(const uint &aUint) noexcept
        requires TwoBytesSizeType<T>;

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <Number T2> DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept; // NOLINT

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2( // NOLINT
        Vector2<T2> &aVec) noexcept
        // Disable the non-const variant for half and bfloat to / from float,
        // otherwise the const specialization will never be picked up:
        requires(!(IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>) &&
                 !(IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>)) &&
                (!std::same_as<T, T2>);

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to BFloat2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept // NOLINT
        requires IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to BFloat2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec, RoundingMode aRoundingMode)
        requires IsBFloat16<T> && IsFloat<T2>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for BFloat2 to float2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept // NOLINT
        requires IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to half2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept // NOLINT
        requires IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for float to half
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec, RoundingMode aRoundingMode)
        requires IsHalfFp16<T> && IsFloat<T2>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for half2 to float2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec) noexcept // NOLINT
        requires IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion for complex with rounding (only for float to bfloat/halffloat)
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector2(const Vector2<T2> &aVec, RoundingMode aRoundingMode)
        requires ComplexFloatingPoint<T> && ComplexFloatingPoint<T2> &&
                 NonNativeFloatingPoint<complex_basetype_t<remove_vector_t<T>>> &&
                 std::same_as<float, complex_basetype_t<remove_vector_t<T2>>>;
    //(std::same_as<Complex<HalfFp16>, T> || std::same_as<Complex<BFloat16>, T>) &&
    //            std::same_as<Complex<float>, T2>;

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector2 FromNV16BitFloat(const nv_bfloat162 &aNVBfloat2) noexcept
        requires IsBFloat16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Vector2 FromNV16BitFloat(const half2 &aNVHalf2) noexcept
        requires IsHalfFp16<T> && CUDA_ONLY<T>;

    ~Vector2() = default;

    Vector2(const Vector2 &) noexcept            = default;
    Vector2(Vector2 &&) noexcept                 = default;
    Vector2 &operator=(const Vector2 &) noexcept = default;
    Vector2 &operator=(Vector2 &&) noexcept      = default;

  private:
    // if we make those converter public we will get in trouble with some T constructors / operators
    /// <summary>
    /// converter to uint for SIMD operations
    /// </summary>
    DEVICE_CODE operator const uint &() const // NOLINT
        requires TwoBytesSizeType<T>;

    /// <summary>
    /// converter to uint for SIMD operations
    /// </summary>
    DEVICE_CODE operator uint &() // NOLINT
        requires TwoBytesSizeType<T>;

    /// <summary>
    /// converter to nv_bfloat162 for SIMD operations
    /// </summary>
    DEVICE_CODE operator const nv_bfloat162 &() const // NOLINT
        requires IsBFloat16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// converter to nv_bfloat162 for SIMD operations
    /// </summary>
    DEVICE_CODE operator nv_bfloat162 &() // NOLINT
        requires IsBFloat16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// converter to half2 for SIMD operations
    /// </summary>
    DEVICE_CODE operator const half2 &() const // NOLINT
        requires IsHalfFp16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// converter to half2 for SIMD operations
    /// </summary>
    DEVICE_CODE operator half2 &() // NOLINT
        requires IsHalfFp16<T> && CUDA_ONLY<T>;

#pragma endregion
  public:
#pragma region Operators
    // don't use space-ship operator as it returns true if any comparison returns true.
    // But NPP only returns true if all channels fulfill the comparison.
    // auto operator<=>(const Vector2 &) const = default;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector2 &aLeft, const Vector2 &aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector2 &aLeft, const Vector2 &aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector2 &aLeft, const Vector2 &aRight, T aEpsilon)
        requires Is16BitFloat<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector2 &aLeft, const Vector2 &aRight,
                                                complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const;

    /// <summary>
    /// Returns true if any element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires IsUShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires IsShort<T> && RealNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>=(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator>(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<=(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator<(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-() const
        requires RealSignedNumber<T> || ComplexNumber<T>;

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-() const
        requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-() const
        requires Is16BitFloat<T> && RealSignedNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(T aOther);

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther);

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator+=(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise addition SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator+(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(T aOther);

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther);

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator-=(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther);

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector2 &SubInv(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const
        requires IsUShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator-(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector2 &operator*=(T aOther);

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector2 &operator*=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector2 &operator*=(const Vector2 &aOther);

    /// <summary>
    /// Component wise multiplication SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator*=(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator*(const Vector2 &aOther) const;

    /// <summary>
    /// Component wise multiplication SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator*(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector2 &operator/=(T aOther);

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector2 &operator/=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector2 &operator/=(const Vector2 &aOther);

    /// <summary>
    /// Component wise division SIMD
    /// </summary>
    DEVICE_CODE Vector2 &operator/=(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise division (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInv(const Vector2 &aOther);

    /// <summary>
    /// Component wise division SIMD (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInv(const Vector2 &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator/(const Vector2 &aOther) const;

    /// <summary>
    /// Component wise division SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2 operator/(const Vector2 &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Inplace Integer division with element wise round()
    /// </summary>
    DEVICE_CODE Vector2 &DivRound(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector2 &DivRoundNearest(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector2 &DivRoundZero(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE Vector2 &DivFloor(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE Vector2 &DivCeil(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvRound(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even (inverted inplace div: this =
    /// aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvRoundNearest(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero (inverted inplace div: this = aOther /
    /// this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvRoundZero(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvFloor(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvCeil(const Vector2 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivRound(const Vector2 &aLeft, const Vector2 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivRoundNearest(const Vector2 &aLeft, const Vector2 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivRoundZero(const Vector2 &aLeft, const Vector2 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivFloor(const Vector2 &aLeft, const Vector2 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivCeil(const Vector2 &aLeft, const Vector2 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleRound(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round nearest ties to even (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleRoundNearest(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round toward zero (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleRoundZero(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise floor (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleFloor(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise ceil() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleCeil(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round()
    /// </summary>
    DEVICE_CODE Vector2 &DivRound(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector2 &DivRoundNearest(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector2 &DivRoundZero(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE Vector2 &DivFloor(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE Vector2 &DivCeil(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvRound(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even (inverted inplace div: this =
    /// aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvRoundNearest(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero (inverted inplace div: this = aOther /
    /// this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvRoundZero(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvFloor(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector2 &DivInvCeil(const Vector2 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivRound(const Vector2 &aLeft, const Vector2 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivRoundNearest(const Vector2 &aLeft, const Vector2 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivRoundZero(const Vector2 &aLeft, const Vector2 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivFloor(const Vector2 &aLeft, const Vector2 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2 DivCeil(const Vector2 &aLeft, const Vector2 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleRound(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round nearest ties to even (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleRoundNearest(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round toward zero (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleRoundZero(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise floor (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleFloor(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise ceil() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector2 &DivScaleCeil(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T &operator[](Axis2D aAxis) const
        requires DeviceCode<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] const T &operator[](Axis2D aAxis) const
        requires HostCode<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] T &operator[](Axis2D aAxis)
        requires DeviceCode<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] T &operator[](Axis2D aAxis)
        requires HostCode<T>;
#pragma endregion

#pragma region Convert Methods
    /// <summary>
    /// Type conversion without saturation, direct type conversion
    /// </summary>
    template <Number T2> [[nodiscard]] static Vector2<T> DEVICE_CODE Convert(const Vector2<T2> &aVec);
#pragma endregion

#pragma region Integral only Methods
    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector2<T> &LShift(const Vector2<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> LShift(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector2<T> &RShift(const Vector2<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RShift(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector2<T> &LShift(uint aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> LShift(const Vector2<T> &aLeft, uint aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector2<T> &RShift(uint aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RShift(const Vector2<T> &aLeft, uint aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE Vector2<T> &And(const Vector2<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> And(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE Vector2<T> &Or(const Vector2<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Or(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE Vector2<T> &Xor(const Vector2<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Xor(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE Vector2<T> &Not()
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Not(const Vector2<T> &aVec)
        requires RealIntegral<T>;
#pragma endregion

#pragma region Methods
#pragma region Exp
    /// <summary>
    /// Element wise exponential
    /// </summary>
    Vector2<T> &Exp()
        requires HostCode<T> && NativeNumber<T>;
    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Exp()
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T>) || ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector2<T> &Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise exponential (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Exp()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise exponential (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Exp(const Vector2<T> &aVec)
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeNumber<T>) || ComplexFloatingPoint<T>;
#pragma endregion

#pragma region Log
    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    Vector2<T> &Ln()
        requires HostCode<T> && NativeNumber<T>;
    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector2<T> &Ln()
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector2<T> &Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Ln()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ln(const Vector2<T> &aVec)
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>;
#pragma endregion

#pragma region Sqr
    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE Vector2<T> &Sqr();

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqr(const Vector2<T> &aVec);
#pragma endregion

#pragma region Sqrt
    /// <summary>
    /// Element wise square root
    /// </summary>
    Vector2<T> &Sqrt()
        requires HostCode<T> && NativeNumber<T>;
    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector2<T> &Sqrt()
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector2<T> &Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise square root (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Sqrt()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise square root (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Sqrt(const Vector2<T> &aVec)
        requires((HostCode<T> || (!EnableSIMD<T>)) && NonNativeFloatingPoint<T>) || ComplexFloatingPoint<T>;
#pragma endregion

#pragma region Abs
    /// <summary>
    /// Element wise absolute
    /// </summary>
    Vector2<T> &Abs()
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    Vector2<T> &Abs()
        requires RealSignedNumber<T> && NonNativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector2<T> &Abs()
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Abs()
        requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Abs()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute  (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires IsShort<T> && RealSignedNumber<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Abs(const Vector2<T> &aVec)
        requires RealSignedNumber<T> && NonNativeNumber<T>;
#pragma endregion

#pragma region AbsDiff
    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector2<T> &AbsDiff(const Vector2<T> &aOther)
        requires RealSignedNumber<T> && NonNativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealSignedNumber<T> && NonNativeNumber<T>;
#pragma endregion

#pragma region Methods for Complex types
    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE Vector2<T> &Conj()
        requires ComplexNumber<T>;

    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Conj(const Vector2<T> &aValue)
        requires ComplexNumber<T>;

    /// <summary>
    /// Conjugate complex multiplication: this * conj(aOther)  per element
    /// </summary>
    DEVICE_CODE Vector2<T> &ConjMul(const Vector2<T> &aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Conjugate complex multiplication: aLeft * conj(aRight) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> ConjMul(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires ComplexNumber<T>;

    /// <summary>
    /// Complex magnitude per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<complex_basetype_t<T>> Magnitude() const
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Complex magnitude squared per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<complex_basetype_t<T>> MagnitudeSqr() const
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Angle between real and imaginary of a complex number (atan2(image, real)) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<complex_basetype_t<T>> Angle() const
        requires ComplexFloatingPoint<T>;
#pragma endregion

#pragma region Clamp
    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector2<T> &Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    Vector2<T> &Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector2<T> &Clamp(T aMinVal, T aMaxVal)
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector2<T> &Clamp(complex_basetype_t<T> aMinVal, complex_basetype_t<T> aMaxVal)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector2<T> &ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>);

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector2<T> &ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>);
#pragma endregion

#pragma region Min
    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aOther)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aOther)
        requires IsUShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    Vector2<T> &Min(const Vector2<T> &aOther)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector2<T> &Min(const Vector2<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>));

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
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
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aOther)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aOther)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aOther)
        requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    Vector2<T> &Max(const Vector2<T> &aOther)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector2<T> &Max(const Vector2<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>) && (HostCode<T> || (DeviceCode<T> && !EnableSIMD<T>));

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsUShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires NonNativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
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
    DEVICE_CODE [[nodiscard]] static Vector2<T> Round(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Round()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Vector2<T> &Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    Vector2<T> &Round()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Floor(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE Vector2<T> &Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    Vector2<T> &Floor()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Floor()
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Floor()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Ceil(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE Vector2<T> &Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    Vector2<T> &Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Ceil()
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &Ceil()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RoundNearest(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector2<T> &RoundNearest()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    Vector2<T> &RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &RoundNearest()
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &RoundNearest()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> RoundZero(const Vector2<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector2<T> &RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    Vector2<T> &RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &RoundZero()
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE Vector2<T> &RoundZero()
        requires NonNativeFloatingPoint<T>;
#pragma endregion

#pragma region Data accessors
    /// <summary>
    /// Provide a smiliar accessor to inner data as for std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] T *data();

    /// <summary>
    /// Provide a smiliar accessor to inner data as for std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T *data() const;
#pragma endregion

#pragma region Compare per element
    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    [[nodiscard]] static Vector2<byte> CompareEQEps(const Vector2<T> &aLeft, const Vector2<T> &aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQEps(const Vector2<T> &aLeft, const Vector2<T> &aRight,
                                                                T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQEps(const Vector2<T> &aLeft, const Vector2<T> &aRight,
                                                                T aEpsilon)
        requires Is16BitFloat<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQEps(const Vector2<T> &aLeft, const Vector2<T> &aRight,
                                                                complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQ(const Vector2<T> &aLeft, const Vector2<T> &aRight);

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareGE(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareGT(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareLE(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareLT(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareNEQ(const Vector2<T> &aLeft, const Vector2<T> &aRight);
#pragma endregion
#pragma endregion
};

template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator+(const Vector2<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft.x + aRight), static_cast<T>(aLeft.y + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator+(T2 aLeft, const Vector2<T> &aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft + aRight.x), static_cast<T>(aLeft + aRight.y)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator-(const Vector2<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft.x - aRight), static_cast<T>(aLeft.y - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator-(T2 aLeft, const Vector2<T> &aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft - aRight.x), static_cast<T>(aLeft - aRight.y)};
}

template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator*(const Vector2<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft.x * aRight), static_cast<T>(aLeft.y * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator*(T2 aLeft, const Vector2<T> &aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft * aRight.x), static_cast<T>(aLeft * aRight.y)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator/(const Vector2<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft.x / aRight), static_cast<T>(aLeft.y / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector2<T> operator/(T2 aLeft, const Vector2<T> &aRight)
    requires Number<T2>
{
    return Vector2<T>{static_cast<T>(aLeft / aRight.x), static_cast<T>(aLeft / aRight.y)};
}

template <HostCode T2> MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<T2> &aVec);

template <HostCode T2> MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<T2> &aVec);

// byte and sbyte are treated as characters and not numbers:
template <HostCode T2>
MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2> MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<T2> &aVec);

template <HostCode T2> MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<T2> &aVec);

template <HostCode T2>
MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<T2> &aVec)
    requires ByteSizeType<T2>;

#ifdef IS_HOST_COMPILER
extern template struct Vector2<sbyte>;
extern template struct Vector2<byte>;
extern template struct Vector2<short>;
extern template struct Vector2<ushort>;
extern template struct Vector2<int>;
extern template struct Vector2<uint>;
extern template struct Vector2<long64>;
extern template struct Vector2<ulong64>;

extern template struct Vector2<BFloat16>;
extern template struct Vector2<HalfFp16>;
extern template struct Vector2<float>;
extern template struct Vector2<double>;

extern template struct Vector2<Complex<sbyte>>;
extern template struct Vector2<Complex<short>>;
extern template struct Vector2<Complex<int>>;
extern template struct Vector2<Complex<long64>>;
extern template struct Vector2<Complex<BFloat16>>;
extern template struct Vector2<Complex<HalfFp16>>;
extern template struct Vector2<Complex<float>>;
extern template struct Vector2<Complex<double>>;

extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<double> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<float> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<double> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<float> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<float> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<sbyte> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<short> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<int> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<long64> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<float>> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<float>> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<float> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<float> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<float>> &) noexcept;

extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<sbyte> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<sbyte> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<sbyte> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<sbyte> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<byte> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<byte> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<byte> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<byte> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<short> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<short> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<short> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<short> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<ushort> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<ushort> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<ushort> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<ushort> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<int> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<int> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<int> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<int> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<uint> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<uint> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<uint> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<uint> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<long64> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<long64> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<long64> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<long64> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<ulong64> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<ulong64> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<ulong64> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<ulong64> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<BFloat16> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<HalfFp16> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<float> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<float> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<float> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<float> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<double> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<double> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<double> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<double> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<sbyte>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<sbyte>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<sbyte>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<sbyte>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<short>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<short>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<short>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<short>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<int>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<int>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<int>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<int>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<long64>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<long64>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<long64>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<long64>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<BFloat16>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<BFloat16>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<BFloat16>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<BFloat16>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<HalfFp16>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<HalfFp16>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<HalfFp16>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<HalfFp16>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<float>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<float>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<float>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<float>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<double>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<double>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<double>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<double>> &aVec);
#endif // IS_HOST_COMPILER
} // namespace mpp