#pragma once
#include "defines.h"
#include "dllexport_common.h"
#include "needSaturationClamp.h"
#include <common/bfloat16.h>
#include <common/half_fp16.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <complex>
#include <concepts>
#include <iostream>
#include <type_traits>

#ifdef IS_CUDA_COMPILER
#include "mpp_defs.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#else
namespace mpp
{
// these types are only used with CUDA, but nevertheless they need
// to be defined, so we set them to some knwon type of same size:
using nv_bfloat162 = int;
using half2        = float;
using float2       = double;
} // namespace mpp
#endif

namespace mpp
{

// forward declaration
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector2;
template <RealSignedNumber T> struct MPPEXPORTFWDDECL_COMMON Complex;

using c_short    = Complex<short>;
using c_int      = Complex<int>;
using c_long     = Complex<long64>;
using c_float    = Complex<float>;
using c_double   = Complex<double>;
using c_HalfFp16 = Complex<HalfFp16>;
using c_BFloat16 = Complex<BFloat16>;

/// <summary>
/// Our own definition of a complex number, that we can use on device and host
/// </summary>
template <RealSignedNumber T> struct alignas(2 * sizeof(T)) MPPEXPORT_COMMON Complex
{
    T real; // NOLINT
    T imag; // NOLINT

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    Complex() noexcept = default;

    /// <summary>
    /// Initializes complex number with only real part, imag = 0
    /// </summary>
    DEVICE_CODE constexpr Complex(T aVal) noexcept : real(aVal), imag(static_cast<T>(0)) // NOLINT
    {
    }

    /// <summary>
    /// Initializes complex number with only real part, imag = 0 (avoid confusion with the array constructor)
    /// </summary>
    DEVICE_CODE constexpr Complex(int aVal) noexcept // NOLINT
        requires(!IsInt<T>)
        : real(static_cast<T>(aVal)), imag(static_cast<T>(0))
    {
    }

    /// <summary>
    /// Initializes complex number with real = aVal[0], imag = aVal[1]
    /// </summary>
    DEVICE_CODE explicit Complex(T aVal[2]) noexcept;

    /// <summary>
    /// Initializes complex number with real = aReal, imag = aImag
    /// </summary>
    DEVICE_CODE Complex(T aReal, T aImag) noexcept; // NOLINT

    /// <summary>
    /// Convert from Vector2 of same type
    /// </summary>
    DEVICE_CODE Complex(const Vector2<T> &aVec) noexcept; // NOLINT

    /// <summary>
    /// Convert from std::complex of same type
    /// </summary>
    Complex(const std::complex<T> &aCplx) noexcept; // NOLINT

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <RealSignedNumber T2>
    DEVICE_CODE Complex( // NOLINT
        const Complex<T2> &aCplx) noexcept
        //  Disable the non-const variant for half and bfloat to / from float,
        //  otherwise the const specialization will never be picked up:
        requires(!(IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>) &&
                 !(IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>));

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <RealSignedNumber T2>
    DEVICE_CODE Complex(Complex<T2> &aVec) noexcept // NOLINT
        requires(!std::same_as<T, T2>);

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to BFloat2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec) noexcept // NOLINT
        requires IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to BFloat2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec, RoundingMode aRoundingMode) noexcept
        requires IsBFloat16<T> && IsFloat<T2>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for BFloat2 to float2
    /// </summary>
    template <IsBFloat16 T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec) noexcept // NOLINT
        requires IsFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to half2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec) noexcept // NOLINT
        requires IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for float to half
    /// </summary>
    template <Number T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec, RoundingMode aRoundingMode) noexcept
        requires IsHalfFp16<T> && IsFloat<T2>;

    /// <summary>
    /// Type conversion using CUDA intrinsics for half2 to float2
    /// </summary>
    template <IsHalfFp16 T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec) noexcept // NOLINT
        requires IsFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Complex FromUint(const uint &aUint) noexcept
        requires TwoBytesSizeType<T>;

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Complex FromNV16BitFloat(const nv_bfloat162 &aNVBfloat2) noexcept
        requires IsBFloat16<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Complex FromNV16BitFloat(const half2 &aNVHalf2) noexcept
        requires IsHalfFp16<T> && CUDA_ONLY<T>;

    ~Complex() = default;

    Complex(const Complex &) noexcept            = default;
    Complex(Complex &&) noexcept                 = default;
    Complex &operator=(const Complex &) noexcept = default;
    Complex &operator=(Complex &&) noexcept      = default;

  private:
    // if we make those converters public we will get in trouble with some T constructors / operators
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
    // No operators for < or > as complex numbers have no ordering

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    [[nodiscard]] static bool EqEps(Complex aLeft, Complex aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Complex aLeft, Complex aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Complex aLeft, Complex aRight, T aEpsilon)
        requires Is16BitFloat<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(const Complex &aOther) const;

    /// <summary>
    /// Returns true if any element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(const Complex &aOther) const;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Complex &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Complex &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Complex &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Complex &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Complex &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Complex &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-() const;

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-() const
        requires IsShort<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-() const
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Complex addition (only real part)
    /// </summary>
    DEVICE_CODE Complex &operator+=(T aOther);

    /// <summary>
    /// Complex addition
    /// </summary>
    DEVICE_CODE Complex &operator+=(const Complex &aOther);

    /// <summary>
    /// Complex addition SIMD
    /// </summary>
    DEVICE_CODE Complex &operator+=(const Complex &aOther)
        requires IsShort<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Complex addition SIMD
    /// </summary>
    DEVICE_CODE Complex &operator+=(const Complex &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Complex addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator+(const Complex &aOther) const;

    /// <summary>
    /// Complex subtraction (only real part)
    /// </summary>
    DEVICE_CODE Complex &operator-=(T aOther);

    /// <summary>
    /// Complex subtraction
    /// </summary>
    DEVICE_CODE Complex &operator-=(const Complex &aOther);

    /// <summary>
    /// Complex subtraction SIMD
    /// </summary>
    DEVICE_CODE Complex &operator-=(const Complex &aOther)
        requires IsShort<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Complex subtraction SIMD
    /// </summary>
    DEVICE_CODE Complex &operator-=(const Complex &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Complex subtraction (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Complex &SubInv(const Complex &aOther);

    /// <summary>
    /// Complex subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Complex &SubInv(const Complex &aOther)
        requires IsShort<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Complex subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Complex &SubInv(const Complex &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Complex subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-(const Complex &aOther) const;

    /// <summary>
    /// Complex subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-(const Complex &aOther) const
        requires IsShort<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Complex subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-(const Complex &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Complex multiplication with real number
    /// </summary>
    DEVICE_CODE Complex &operator*=(T aOther);

    /// <summary>
    /// Complex multiplication
    /// </summary>
    DEVICE_CODE Complex &operator*=(const Complex &aOther);

    /// <summary>
    /// Complex multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator*(const Complex &aOther) const;

    /// <summary>
    /// Complex division with real number
    /// </summary>
    DEVICE_CODE Complex &operator/=(T aOther);

    /// <summary>
    /// Complex division
    /// </summary>
    DEVICE_CODE Complex &operator/=(const Complex &aOther);

    /// <summary>
    /// Complex division (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Complex &DivInv(const Complex &aOther);

    /// <summary>
    /// Complex division
    /// </summary>
    DEVICE_CODE Complex operator/(const Complex &aOther) const;

    /// <summary>
    /// Inplace complex integer division with element wise round()
    /// </summary>
    DEVICE_CODE Complex &DivRound(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Complex &DivRoundNearest(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE Complex &DivRoundZero(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise floor()
    /// </summary>
    DEVICE_CODE Complex &DivFloor(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE Complex &DivCeil(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise round() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Complex &DivInvRound(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise round nearest ties to even (inverted inplace div: this =
    /// aOther / this)
    /// </summary>
    DEVICE_CODE Complex &DivInvRoundNearest(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise round toward zero (inverted inplace div: this = aOther /
    /// this)
    /// </summary>
    DEVICE_CODE Complex &DivInvRoundZero(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise floor() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Complex &DivInvFloor(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise ceil() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Complex &DivInvCeil(const Complex &aOther)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Complex integer division with element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex DivRound(const Complex &aLeft, const Complex &aRight)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Complex integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex DivRoundNearest(const Complex &aLeft, const Complex &aRight)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Complex integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex DivRoundZero(const Complex &aLeft, const Complex &aRight)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Complex integer division with element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex DivFloor(const Complex &aLeft, const Complex &aRight)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Complex integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex DivCeil(const Complex &aLeft, const Complex &aRight)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise round() (for scaling operations)
    /// </summary>
    DEVICE_CODE Complex &DivScaleRound(T aScale)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise round nearest ties to even (for scaling operations)
    /// </summary>
    DEVICE_CODE Complex &DivScaleRoundNearest(T aScale)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise round toward zero (for scaling operations)
    /// </summary>
    DEVICE_CODE Complex &DivScaleRoundZero(T aScale)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise floor (for scaling operations)
    /// </summary>
    DEVICE_CODE Complex &DivScaleFloor(T aScale)
        requires RealSignedIntegral<T>;

    /// <summary>
    /// Inplace complex integer division with element wise ceil() (for scaling operations)
    /// </summary>
    DEVICE_CODE Complex &DivScaleCeil(T aScale)
        requires RealSignedIntegral<T>;
#pragma endregion

#pragma region Methods
#pragma region Exp
    /// <summary>
    /// Complex exponential
    /// </summary>
    Complex &Exp()
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Complex exponential
    /// </summary>
    Complex &Exp()
        requires HostCode<T> && Is16BitFloat<T>;

    /// <summary>
    /// Complex exponential
    /// </summary>
    DEVICE_CODE Complex &Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>;
    /// <summary>
    /// Complex exponential
    /// </summary>
    DEVICE_CODE Complex &Exp()
        requires DeviceCode<T> && Is16BitFloat<T>;

    /// <summary>
    /// Complex exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex Exp(const Complex &aVec);
#pragma endregion

#pragma region Log
    /// <summary>
    /// Complex natural logarithm
    /// </summary>
    Complex &Ln()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Complex natural logarithm
    /// </summary>
    DEVICE_CODE Complex &Ln()
        requires NonNativeNumber<T>;

    /// <summary>
    /// Complex natural logarithm
    /// </summary>
    DEVICE_CODE Complex &Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Complex natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex Ln(const Complex &aVec)
        requires RealFloatingPoint<T>;
#pragma endregion

#pragma region Sqr
    /// <summary>
    /// Complex square
    /// </summary>
    DEVICE_CODE Complex &Sqr();

    /// <summary>
    /// Complex square
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex Sqr(const Complex &aVec);
#pragma endregion

#pragma region Sqrt
    /// <summary>
    /// Complex square root
    /// </summary>
    Complex &Sqrt()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Complex square root
    /// </summary>
    DEVICE_CODE Complex &Sqrt()
        requires NonNativeNumber<T>;

    /// <summary>
    /// Complex square root
    /// </summary>
    DEVICE_CODE Complex &Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Complex square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex Sqrt(const Complex &aVec)
        requires RealFloatingPoint<T>;
#pragma endregion

    /// <summary>
    /// Conjugate complex
    /// </summary>
    DEVICE_CODE Complex<T> &Conj();

    /// <summary>
    /// Conjugate complex
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Conj(const Complex<T> &aValue);

    /// <summary>
    /// Conjugate complex multiplication: this * conj(aOther)
    /// </summary>
    DEVICE_CODE Complex<T> &ConjMul(const Complex<T> &aOther);

    /// <summary>
    /// Conjugate complex multiplication: aLeft * conj(aRight)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> ConjMul(const Complex<T> &aLeft, const Complex<T> &aRight);

    /// <summary>
    /// Complex magnitude |a+bi|
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Complex magnitude |a+bi|
    /// </summary>
    [[nodiscard]] T Magnitude() const
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Complex magnitude |a+bi|
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires NonNativeNumber<T>;

    /// <summary>
    /// Complex magnitude squared |a+bi|^2
    /// </summary>
    DEVICE_CODE [[nodiscard]] T MagnitudeSqr() const
        requires RealFloatingPoint<T>;

    /// <summary>
    /// Angle between real and imaginary of a complex number (atan2(image, real))
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Angle() const
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Angle between real and imaginary of a complex number (atan2(image, real))
    /// </summary>
    [[nodiscard]] T Angle() const
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Angle between real and imaginary of a complex number (atan2(image, real))
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Angle() const
        requires NonNativeNumber<T>;

    /// <summary>
    /// Complex clamp to value range
    /// </summary>
    DEVICE_CODE Complex<T> &Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Complex clamp to value range
    /// </summary>
    Complex<T> &Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Complex clamp to value range
    /// </summary>
    DEVICE_CODE Complex<T> &Clamp(T aMinVal, T aMaxVal)
        requires NonNativeNumber<T>;

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Complex<T> &ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>);

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Complex<T> &ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>);

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Round(const Complex<T> &aValue)
        requires RealFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Complex<T> &Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    Complex<T> &Round()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_ONLY_CODE Complex<T> &Round()
        requires NonNativeNumber<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Floor(const Complex<T> &aValue)
        requires RealFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE Complex<T> &Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    Complex<T> &Floor()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Complex<T> &Floor()
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE Complex<T> &Floor()
        requires NonNativeNumber<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Ceil(const Complex<T> &aValue)
        requires RealFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE Complex<T> &Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    Complex<T> &Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Complex<T> &Ceil()
        requires Is16BitFloat<T> && CUDA_ONLY<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE Complex<T> &Ceil()
        requires NonNativeNumber<T>;

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> RoundNearest(const Complex<T> &aValue)
        requires RealFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Complex<T> &RoundNearest()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    Complex<T> &RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Complex<T> &RoundNearest()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE Complex<T> &RoundNearest()
        requires NonNativeNumber<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> RoundZero(const Complex<T> &aValue)
        requires RealFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE Complex<T> &RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    Complex<T> &RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE Complex<T> &RoundZero()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE Complex<T> &RoundZero()
        requires NonNativeNumber<T>;
};

template <typename T, typename T2>
DEVICE_CODE Complex<T> operator+(const Complex<T> &aLeft, T2 aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{static_cast<T>(aLeft.real + aRight), aLeft.imag};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator+(T2 aLeft, const Complex<T> &aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{static_cast<T>(aLeft + aRight.real), aRight.imag};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator-(const Complex<T> &aLeft, T2 aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{static_cast<T>(aLeft.real - aRight), aLeft.imag};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator-(T2 aLeft, const Complex<T> &aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{static_cast<T>(aLeft - aRight.real), static_cast<T>(-aRight.imag)};
}

template <typename T, typename T2>
DEVICE_CODE Complex<T> operator*(const Complex<T> &aLeft, T2 aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{static_cast<T>(aLeft.real * aRight), static_cast<T>(aLeft.imag * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator*(T2 aLeft, const Complex<T> &aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{static_cast<T>(aLeft * aRight.real), static_cast<T>(aLeft * aRight.imag)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator/(const Complex<T> &aLeft, T2 aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{static_cast<T>(aLeft.real / aRight), static_cast<T>(aLeft.imag / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator/(T2 aLeft, const Complex<T> &aRight)
    requires RealSignedNumber<T2>
{
    const T leftReal = static_cast<T>(aLeft);
    const T denom    = aRight.real * aRight.real + aRight.imag * aRight.imag;
    if constexpr (RealIntegral<T>)
    { // floats will denormalize to inf, but likely just an overflow
        if (denom == T(0))
        {
            return {static_cast<T>(0), static_cast<T>(0)};
        }
    }
    const T tempReal = leftReal * aRight.real;
    const T tempImag = -leftReal * aRight.imag;
    return {static_cast<T>(tempReal / denom), static_cast<T>(tempImag / denom)}; // complex division
}

// complex literal: gives a complex number with real part 0 and imaginary part being the number before _i when mpp
// namespace is used
inline Complex<float> operator""_i(long double aValue)
{
    return {0.0f, static_cast<float>(aValue)};
}

// complex literal: gives a complex number with real part 0 and imaginary part being the number before _i when mpp
// namespace is used
inline Complex<int> operator""_i(unsigned long long int aValue)
{
    return {0, static_cast<int>(aValue)};
}

// complex literal: gives a complex number with real part 0 and imaginary part being the number before _ih when mpp
// namespace is used (HalfFp16-complex)
inline Complex<HalfFp16> operator""_ih(long double aValue)
{
    return {static_cast<HalfFp16>(0.0f), static_cast<HalfFp16>(static_cast<float>(aValue))};
}

// complex literal: gives a complex number with real part 0 and imaginary part being the number before _ih when mpp
// namespace is used (HalfFp16-complex)
inline Complex<BFloat16> operator""_ib(long double aValue)
{
    return {static_cast<BFloat16>(0.0f), static_cast<BFloat16>(static_cast<float>(aValue))};
}

template <mpp::HostCode T2> MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<T2> &aVec);
template <mpp::HostCode T2>
MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<T2> &aVec);
template <mpp::HostCode T2> MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<T2> &aVec);
template <mpp::HostCode T2> MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<T2> &aVec);

template <typename T> struct make_complex
{
    // default to int so that we get a valid Complex type in function defintions (otherwise it wouldn't compile)
    using type = Complex<int>;
};
template <> struct make_complex<sbyte>
{
    using type = Complex<sbyte>;
};
template <> struct make_complex<short>
{
    using type = Complex<short>;
};
template <> struct make_complex<int>
{
    using type = Complex<int>;
};
template <> struct make_complex<HalfFp16>
{
    using type = Complex<HalfFp16>;
};
template <> struct make_complex<BFloat16>
{
    using type = Complex<BFloat16>;
};
template <> struct make_complex<float>
{
    using type = Complex<float>;
};
template <> struct make_complex<double>
{
    using type = Complex<double>;
};

template <typename T> using make_complex_t = typename make_complex<T>::type;

template <RealFloatingPoint T> DEVICE_CODE inline bool isnan(const Complex<T> &aVal)
{
    return isnan(aVal.real) || isnan(aVal.imag);
}
template <RealFloatingPoint T> DEVICE_CODE inline bool isinf(const Complex<T> &aVal)
{
    return isinf(aVal.real) || isinf(aVal.imag);
}
template <RealFloatingPoint T> DEVICE_CODE inline bool isfinite(const Complex<T> &aVal)
{
    return isfinite(aVal.real) && isfinite(aVal.imag);
}

#ifdef IS_HOST_COMPILER
extern template struct Complex<sbyte>;
extern template struct Complex<short>;
extern template struct Complex<int>;
extern template struct Complex<long64>;
extern template struct Complex<float>;
extern template struct Complex<double>;
extern template struct Complex<HalfFp16>;
extern template struct Complex<BFloat16>;

extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(const Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(const Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(const Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(const Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(const Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(const Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(const Complex<float> &, RoundingMode) noexcept;

extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(const Complex<float> &, RoundingMode) noexcept;

extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<sbyte>::Complex(Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<short>::Complex(Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<short>::Complex(Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<int>::Complex(Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<int>::Complex(Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<long64>::Complex(Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<float>::Complex(Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<float>::Complex(Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<double>::Complex(Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(Complex<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Complex<double>::Complex(Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<HalfFp16>::Complex(Complex<BFloat16> &) noexcept;

extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<short> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<int> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<long64> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<float> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<double> &) noexcept;
extern template MPPEXPORT_COMMON Complex<BFloat16>::Complex(Complex<HalfFp16> &) noexcept;

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<mpp::sbyte> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<mpp::sbyte> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<mpp::sbyte> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<mpp::sbyte> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<short> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<short> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<short> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<short> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<long64> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<long64> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<long64> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<long64> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<int> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<int> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<int> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<int> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<float> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<float> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<float> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<float> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<double> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<double> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<double> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<double> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<mpp::HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<mpp::HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<mpp::HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<mpp::HalfFp16> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const mpp::Complex<mpp::BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const mpp::Complex<mpp::BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, mpp::Complex<mpp::BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, mpp::Complex<mpp::BFloat16> &aVec);
#endif
} // namespace mpp