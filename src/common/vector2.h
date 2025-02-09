#pragma once
#include "bfloat16.h"
#include "complex.h"
#include "defines.h"
#include "half_fp16.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "opp_defs.h"
#include "vector_typetraits.h"
#include <concepts>
#include <iostream>
#include <type_traits>

namespace opp
{

// forward declaration:
template <Number T> struct Vector1;
template <Number T> struct Vector2;
template <Number T> struct Vector3;
template <Number T> struct Vector4;

enum class Axis2D // NOLINT(performance-enum-size)
{
    X = 0,
    Y = 1
};

std::ostream &operator<<(std::ostream &aOs, const Axis2D &aAxis);
std::wostream &operator<<(std::wostream &aOs, const Axis2D &aAxis);

/// <summary>
/// A two T component vector. Can replace CUDA's vector2 types
/// </summary>
template <Number T> struct alignas(2 * sizeof(T)) Vector2
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
    DEVICE_CODE Vector2(const Vector2<T2> &aVec, RoundingMode aRoundingMode) noexcept
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
    DEVICE_CODE Vector2(const Vector2<T2> &aVec, RoundingMode aRoundingMode) noexcept
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
    DEVICE_CODE Vector2(const Vector2<T2> &aVec, RoundingMode aRoundingMode) noexcept
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
    DEVICE_CODE [[nodiscard]] static bool EqEps(Vector2 aLeft, Vector2 aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Vector2 aLeft, Vector2 aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Vector2 aLeft, Vector2 aRight, T aEpsilon)
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
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires HostCode<T> && NativeFloatingPoint<T>;

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
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> AbsDiff(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

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
    [[nodiscard]] static Vector2<byte> CompareEQEps(Vector2<T> aLeft, Vector2<T> aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQEps(Vector2<T> aLeft, Vector2<T> aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<byte> CompareEQEps(Vector2<T> aLeft, Vector2<T> aRight, T aEpsilon)
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

template <HostCode T2> std::ostream &operator<<(std::ostream &aOs, const Vector2<T2> &aVec);

template <HostCode T2> std::wostream &operator<<(std::wostream &aOs, const Vector2<T2> &aVec);

// byte and sbyte are treated as characters and not numbers:
template <HostCode T2>
std::ostream &operator<<(std::ostream &aOs, const Vector2<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
std::wostream &operator<<(std::wostream &aOs, const Vector2<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2> std::istream &operator>>(std::istream &aIs, Vector2<T2> &aVec);

template <HostCode T2> std::wistream &operator>>(std::wistream &aIs, Vector2<T2> &aVec);

template <HostCode T2>
std::istream &operator>>(std::istream &aIs, Vector2<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
std::wistream &operator>>(std::wistream &aIs, Vector2<T2> &aVec)
    requires ByteSizeType<T2>;

} // namespace opp