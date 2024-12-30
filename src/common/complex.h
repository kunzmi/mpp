#pragma once
#include "defines.h"
#include "exception.h"
#include "needSaturationClamp.h"
#include "safeCast.h"
#include <cmath>
#include <common/numberTypes.h>
#include <complex>
#include <concepts>
#include <iostream>
#include <type_traits>

#ifdef IS_CUDA_COMPILER
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#else
namespace opp
{
// these types are only used with CUDA, but nevertheless they need
// to be defined, so we set them to some knwon type of same size:
using nv_bfloat162 = int;
using half2        = float;
using float2       = double;
} // namespace opp

// no arguments to these intrinsics directly depend on a template parameter,
// so a declaration must be available:
opp::float2 __half22float2(opp::half2);
opp::half2 __float22half2_rn(opp::float2);
opp::float2 __bfloat1622float2(opp::nv_bfloat162);
opp::nv_bfloat162 __float22bfloat162_rn(opp::float2);
#endif

namespace opp
{

// forward declaration
template <Number T> struct Vector2;

/// <summary>
/// Our own definition of a complex number, that we can use on device and host
/// </summary>
template <RealSignedNumber T> struct alignas(2 * sizeof(T)) Complex
{
    T real;
    T imag;

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    DEVICE_CODE Complex() noexcept
    {
    }

    /// <summary>
    /// Initializes complex number with only real part, imag = 0
    /// </summary>
    DEVICE_CODE Complex(T aVal) noexcept : real(aVal), imag(0)
    {
    }

    /// <summary>
    /// Initializes complex number with real = aVal[0], imag = aVal[1]
    /// </summary>
    DEVICE_CODE Complex(T aVal[2]) noexcept : real(aVal[0]), imag(aVal[1])
    {
    }

    /// <summary>
    /// Initializes complex number with real = aReal, imag = aImag
    /// </summary>
    DEVICE_CODE Complex(T aReal, T aImag) noexcept : real(aReal), imag(aImag)
    {
    }

    /// <summary>
    /// Convert from Vector2 of same type
    /// </summary>
    DEVICE_CODE Complex(const Vector2<T> &aVec) noexcept : real(aVec.x), imag(aVec.y)
    {
    }

    /// <summary>
    /// Convert from std::complex of same type
    /// </summary>
    Complex(const std::complex<T> &aCplx) noexcept : real(aCplx.real()), imag(aCplx.imag())
    {
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <RealSignedNumber T2>
    DEVICE_CODE Complex(const Complex<T2> &aCplx) noexcept
        // Disable the non-const variant for half and bfloat to / from float,
        // otherwise the const specialization will never be picked up:
        requires(!(IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>) &&
                 !(IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>))
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            Complex<T2> temp(aCplx);
            temp.template ClampToTargetType<T>();
            real = static_cast<T>(temp.real);
            imag = static_cast<T>(temp.imag);
        }
        else
        {
            real = static_cast<T>(aCplx.real);
            imag = static_cast<T>(aCplx.imag);
        }
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <RealSignedNumber T2> DEVICE_CODE Complex(Complex<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            aVec.template ClampToTargetType<T>();
        }
        real = static_cast<T>(aVec.real);
        imag = static_cast<T>(aVec.imag);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to BFloat2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec) noexcept
        requires IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>
    {
        const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
        nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
        *thisPtr              = __float22bfloat162_rn(*aVecPtr);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for BFloat2 to float2
    /// </summary>
    template <IsBFloat16 T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec) noexcept
        requires IsFloat<T> && CUDA_ONLY<T>
    {
        const nv_bfloat162 *aVecPtr = reinterpret_cast<const nv_bfloat162 *>(&aVec);
        float2 *thisPtr             = reinterpret_cast<float2 *>(this);
        *thisPtr                    = __bfloat1622float2(*aVecPtr);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for float2 to half2
    /// </summary>
    template <Number T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec) noexcept
        requires IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>
    {
        const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
        half2 *thisPtr        = reinterpret_cast<half2 *>(this);
        *thisPtr              = __float22half2_rn(*aVecPtr);
    }

    /// <summary>
    /// Type conversion using CUDA intrinsics for half2 to float2
    /// </summary>
    template <IsHalfFp16 T2>
    DEVICE_CODE Complex(const Complex<T2> &aVec) noexcept
        requires IsFloat<T> && CUDA_ONLY<T>
    {
        const half2 *aVecPtr = reinterpret_cast<const half2 *>(&aVec);
        float2 *thisPtr      = reinterpret_cast<float2 *>(this);
        *thisPtr             = __half22float2(*aVecPtr);
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Complex FromUint(const uint &aUint) noexcept
        requires TwoBytesSizeType<T>
    {
        return Complex(*reinterpret_cast<const Complex<T> *>(&aUint));
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Complex FromNV16BitFloat(const nv_bfloat162 &aNVBfloat2) noexcept
        requires IsBFloat16<T> && CUDA_ONLY<T>
    {
        return Complex(*reinterpret_cast<const Complex<T> *>(&aNVBfloat2));
    }

    /// <summary>
    /// Usefull constructor for SIMD instructions
    /// </summary>
    DEVICE_CODE static Complex FromNV16BitFloat(const half2 &aNVHalf2) noexcept
        requires IsHalfFp16<T> && CUDA_ONLY<T>
    {
        return Complex(*reinterpret_cast<const Complex<T> *>(&aNVHalf2));
    }

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
    DEVICE_CODE operator const uint &() const
        requires TwoBytesSizeType<T>
    {
        return *reinterpret_cast<const uint *>(this);
    }

    /// <summary>
    /// converter to uint for SIMD operations
    /// </summary>
    DEVICE_CODE operator uint &()
        requires TwoBytesSizeType<T>
    {
        return *reinterpret_cast<uint *>(this);
    }

    /// <summary>
    /// converter to nv_bfloat162 for SIMD operations
    /// </summary>
    DEVICE_CODE operator const nv_bfloat162 &() const
        requires IsBFloat16<T> && CUDA_ONLY<T>
    {
        return *reinterpret_cast<const nv_bfloat162 *>(this);
    }

    /// <summary>
    /// converter to nv_bfloat162 for SIMD operations
    /// </summary>
    DEVICE_CODE operator nv_bfloat162 &()
        requires IsBFloat16<T> && CUDA_ONLY<T>
    {
        return *reinterpret_cast<nv_bfloat162 *>(this);
    }

    /// <summary>
    /// converter to half2 for SIMD operations
    /// </summary>
    DEVICE_CODE operator const half2 &() const
        requires IsHalfFp16<T> && CUDA_ONLY<T>
    {
        return *reinterpret_cast<const half2 *>(this);
    }

    /// <summary>
    /// converter to half2 for SIMD operations
    /// </summary>
    DEVICE_CODE operator half2 &()
        requires IsHalfFp16<T> && CUDA_ONLY<T>
    {
        return *reinterpret_cast<half2 *>(this);
    }
#pragma endregion
  public:
#pragma region Operators
    // No operators for < or > as complex numbers have no ordering

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(const Complex &aOther) const
    {
        bool res = real == aOther.real;
        res &= imag == aOther.imag;
        return res;
    }

    /// <summary>
    /// Returns true if any element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(const Complex &aOther) const
    {
        bool res = real != aOther.real;
        res |= imag != aOther.imag;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Complex &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpeq2(*this, aOther) == 0xFFFFFFFFU;
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Complex &aOther) const
        requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __vcmpne2(*this, aOther) != 0U;
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Complex &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbeq2(*this, aOther);
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Complex &aOther) const
        requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        // __hbne2 returns true only if both elements are != but we need true if any element is !=
        // so we use hbeq and negate the result
        return !(__hbeq2(*this, aOther));
    }

    /// <summary>
    /// Returns true if each element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator==(const Complex &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return __hbeq2(*this, aOther);
    }

    /// <summary>
    /// Returns true if any element comparison is true (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] bool operator!=(const Complex &aOther) const
        requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        // __hbne2 returns true only if both elements are != but we need true if any element is !=
        // so we use hbeq and negate the result
        return !(__hbeq2(*this, aOther));
    }

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-() const
    {
        return Complex<T>(T(-real), T(-imag));
    }

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-() const
        requires IsShort<T> && CUDA_ONLY<T>
    {
        return FromUint(__vnegss2(*this));
    }

    /// <summary>
    /// Negation (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-() const
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        return FromNV16BitFloat(__hneg2(*this));
    }

    /// <summary>
    /// Complex addition (only real part)
    /// </summary>
    DEVICE_CODE Complex &operator+=(T aOther)
    {
        real += aOther;
        return *this;
    }

    /// <summary>
    /// Complex addition
    /// </summary>
    DEVICE_CODE Complex &operator+=(const Complex &aOther)
    {
        real += aOther.real;
        imag += aOther.imag;
        return *this;
    }

    /// <summary>
    /// Complex addition SIMD
    /// </summary>
    DEVICE_CODE Complex &operator+=(const Complex &aOther)
        requires IsShort<T> && CUDA_ONLY<T>
    {
        *this = FromUint(__vaddss2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Complex addition SIMD
    /// </summary>
    DEVICE_CODE Complex &operator+=(const Complex &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(__hadd2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Complex addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator+(const Complex &aOther) const
    {
        return Complex<T>{T(real + aOther.real), T(imag + aOther.imag)};
    }

    /// <summary>
    /// Complex subtraction (only real part)
    /// </summary>
    DEVICE_CODE Complex &operator-=(T aOther)
    {
        real -= aOther;
        return *this;
    }

    /// <summary>
    /// Complex subtraction
    /// </summary>
    DEVICE_CODE Complex &operator-=(const Complex &aOther)
    {
        real -= aOther.real;
        imag -= aOther.imag;
        return *this;
    }

    /// <summary>
    /// Complex subtraction SIMD
    /// </summary>
    DEVICE_CODE Complex &operator-=(const Complex &aOther)
        requires IsShort<T> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubss2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Complex subtraction SIMD
    /// </summary>
    DEVICE_CODE Complex &operator-=(const Complex &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(__hsub2(*this, aOther));
        return *this;
    }

    /// <summary>
    /// Complex subtraction (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Complex &SubInv(const Complex &aOther)
    {
        real = aOther.real - real;
        imag = aOther.imag - imag;
        return *this;
    }

    /// <summary>
    /// Complex subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Complex &SubInv(const Complex &aOther)
        requires IsShort<T> && CUDA_ONLY<T>
    {
        *this = FromUint(__vsubss2(aOther, *this));
        return *this;
    }

    /// <summary>
    /// Complex subtraction SIMD (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Complex &SubInv(const Complex &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(__hsub2(aOther, *this));
        return *this;
    }

    /// <summary>
    /// Complex subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-(const Complex &aOther) const
    {
        return Complex<T>{T(real - aOther.real), T(imag - aOther.imag)};
    }

    /// <summary>
    /// Complex subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-(const Complex &aOther) const
        requires IsShort<T> && CUDA_ONLY<T>
    {
        return FromUint(__vsubss2(*this, aOther));
    }

    /// <summary>
    /// Complex subtraction SIMD
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-(const Complex &aOther) const
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        return FromNV16BitFloat(__hsub2(*this, aOther));
    }

    /// <summary>
    /// Complex multiplication with real number
    /// </summary>
    DEVICE_CODE Complex &operator*=(T aOther)
    {
        real *= aOther;
        imag *= aOther;
        return *this;
    }

    /// <summary>
    /// Complex multiplication
    /// </summary>
    DEVICE_CODE Complex &operator*=(const Complex &aOther)
    {
        T tempReal = real * aOther.real - imag * aOther.imag;
        T tempImag = real * aOther.imag + imag * aOther.real;
        real       = tempReal;
        imag       = tempImag;
        return *this;
    }

    /// <summary>
    /// Complex multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator*(const Complex &aOther) const
    {
        T tempReal = real * aOther.real - imag * aOther.imag;
        T tempImag = real * aOther.imag + imag * aOther.real;
        return Complex<T>{tempReal, tempImag};
    }

    /// <summary>
    /// Complex division with real number
    /// </summary>
    DEVICE_CODE Complex &operator/=(T aOther)
    {
        real /= aOther;
        imag /= aOther;
        return *this;
    }

    /// <summary>
    /// Complex division
    /// </summary>
    DEVICE_CODE Complex &operator/=(const Complex &aOther)
    {
        T denom    = aOther.real * aOther.real + aOther.imag * aOther.imag;
        T tempReal = real * aOther.real + imag * aOther.imag;
        T tempImag = imag * aOther.real - real * aOther.imag;
        real       = tempReal / denom;
        imag       = tempImag / denom;
        return *this;
    }

    /// <summary>
    /// Complex division (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Complex &DivInv(const Complex &aOther)
    {
        T denom    = real * real + imag * imag;
        T tempReal = aOther.real * real + aOther.imag * imag;
        T tempImag = aOther.imag * real - aOther.real * imag;
        real       = tempReal / denom;
        imag       = tempImag / denom;
        return *this;
    }

    /// <summary>
    /// Complex division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator/(const Complex &aOther) const
    {
        T denom    = aOther.real * aOther.real + aOther.imag * aOther.imag;
        T tempReal = real * aOther.real + imag * aOther.imag;
        T tempImag = imag * aOther.real - real * aOther.imag;
        return Complex<T>{T(tempReal / denom), T(tempImag / denom)};
    }
#pragma endregion

#pragma region Methods
    /// <summary>
    /// Conjugate complex
    /// </summary>
    DEVICE_CODE void Conj()
    {
        imag = -imag;
    }

    /// <summary>
    /// Conjugate complex
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Conj(const Complex<T> &aValue)
    {
        return {aValue.real, -aValue.imag};
    }

    /// <summary>
    /// Conjugate complex multiplication: this * conj(aOther)
    /// </summary>
    DEVICE_CODE void ConjMul(const Complex<T> &aOther)
    {
        const T realTemp = (real * aOther.real) + (imag * aOther.imag);
        const T imagTemp = (aOther.real * imag) - (aOther.imag * real);
        real             = realTemp;
        imag             = imagTemp;
    }

    /// <summary>
    /// Conjugate complex multiplication: aLeft * conj(aRight)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> ConjMul(const Complex<T> &aLeft, const Complex<T> &aRight)
    {
        const T realTemp = (aLeft.real * aRight.real) + (aLeft.imag * aRight.imag);
        const T imagTemp = (aRight.real * aLeft.imag) - (aRight.imag * aLeft.real);

        return {realTemp, imagTemp};
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Complex<T> &aOther)
        requires HostCode<T> && NativeNumber<T>
    {
        real = std::abs(real - aOther.real);
        imag = std::abs(imag - aOther.imag);
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> AbsDiff(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        Complex<T> ret;
        ret.real = std::abs(aLeft.real - aRight.real);
        ret.imag = std::abs(aLeft.imag - aRight.imag);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference (SIMD)
    /// </summary>
    DEVICE_CODE void AbsDiff(const Complex<T> &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T>
    {
        *this = FromUint(__vabsdiffs2(*this, aOther));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Complex<T> &aOther)
        requires DeviceCode<T> && NativeNumber<T>
    {
        real = abs(real - aOther.real);
        imag = abs(imag - aOther.imag);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> AbsDiff(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        Complex<T> ret;
        ret.real = abs(aLeft.real - aRight.real);
        ret.imag = abs(aLeft.imag - aRight.imag);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> AbsDiff(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T>
    {
        return FromUint(__vabsdiffs2(aLeft, aRight));
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Complex<T> &aOther)
        requires NonNativeNumber<T>
    {
        real = T::Abs(real - aOther.real);
        imag = T::Abs(imag - aOther.imag);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> AbsDiff(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires NonNativeNumber<T>
    {
        Complex<T> ret;
        ret.real = T::Abs(aLeft.real - aRight.real);
        ret.imag = T::Abs(aLeft.imag - aRight.imag);
        return ret;
    }

    /// <summary>
    /// Complex magnitude
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        return sqrt(MagnitudeSqr());
    }

    /// <summary>
    /// Complex magnitude
    /// </summary>
    [[nodiscard]] T Magnitude() const
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        return std::sqrt(MagnitudeSqr());
    }

    /// <summary>
    /// Complex magnitude
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires NonNativeNumber<T>
    {
        return T::Sqrt(MagnitudeSqr());
    }

    /// <summary>
    /// Complex magnitude squared
    /// </summary>
    DEVICE_CODE [[nodiscard]] T MagnitudeSqr() const
        requires RealFloatingPoint<T>
    {
        return real * real + imag * imag;
    }

    /// <summary>
    /// Complex clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeNumber<T>
    {
        real = max(aMinVal, min(real, aMaxVal));
        imag = max(aMinVal, min(imag, aMaxVal));
    }

    /// <summary>
    /// Complex clamp to value range
    /// </summary>
    void Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeNumber<T>
    {
        real = std::max(aMinVal, std::min(real, aMaxVal));
        imag = std::max(aMinVal, std::min(imag, aMaxVal));
    }

    /// <summary>
    /// Complex clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires NonNativeNumber<T>
    {
        real = T::Max(aMinVal, T::Min(real, aMaxVal));
        imag = T::Max(aMinVal, T::Min(imag, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>) && (!IsHalfFp16<T> || !IsShort<TTarget>) &&
                (!IsBFloat16<T> || !IsShort<TTarget>)
    {
        Clamp(T(numeric_limits<TTarget>::lowest()), T(numeric_limits<TTarget>::max()));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>) && IsHalfFp16<T> && IsShort<TTarget>
    {
        // special case for half floats: the maximum value of short is slightly larger than the closest exact
        // integer in HalfFp16, and as we use round to nearest, the clamping would result in a too large number.
        // Thus for HalfFp16 and short, we clamp to the exact integer smaller than short::max(), i.e. 32752
        constexpr HalfFp16 maxExactShort = HalfFp16::FromUShort(0x77FF); // = 32752
        Clamp(T(numeric_limits<TTarget>::lowest()), maxExactShort);
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>) && IsBFloat16<T> && IsShort<TTarget>
    {
        // special case for half floats: the maximum value of short is slightly smaller than the closest exact
        // integer in BFloat16, and as we use round to nearest, the clamping would result in a too large number.
        // Thus for BFloat16 and short, we clamp to the next integer smaller than short::max(), i.e. 32640
        constexpr BFloat16 maxExactShort = BFloat16::FromUShort(0x46FF); // = 32640
        Clamp(T(numeric_limits<TTarget>::lowest()), maxExactShort);
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>)
    {
    }

    /// <summary>
    /// Complex minimum
    /// </summary>
    [[nodiscard]] Complex<T> Min(const Complex<T> &aRight) const
        requires HostCode<T> && NativeNumber<T>
    {
        return Complex<T>{std::min(real, aRight.real), std::min(imag, aRight.imag)};
    }

    /// <summary>
    /// Complex minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex<T> Min(const Complex<T> &aRight) const
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Complex<T>{T(min(real, aRight.real)), T(min(imag, aRight.imag))};
    }

    /// <summary>
    /// Complex minimum
    /// </summary>
    DEVICE_CODE void Min(const Complex<T> &aRight)
        requires NonNativeNumber<T>
    {
        real.Min(aRight.real);
        imag.Min(aRight.imag);
    }

    /// <summary>
    /// Complex minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Min(const Complex<T> &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        *this = FromUint(__vmins2(*this, aOther));
    }

    /// <summary>
    /// Complex minimum (SIMD)
    /// </summary>
    DEVICE_CODE void Min(const Complex<T> &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hmin2(*this, aOther));
    }

    /// <summary>
    /// Complex minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Min(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Complex<T>{T(min(aLeft.real, aRight.real)), T(min(aLeft.imag, aRight.imag))};
    }

    /// <summary>
    /// Complex minimum
    /// </summary>
    [[nodiscard]] static Complex<T> Min(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        return Complex<T>{std::min(aLeft.real, aRight.real), std::min(aLeft.imag, aRight.imag)};
    }

    /// <summary>
    /// Complex minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Min(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires NonNativeNumber<T>
    {
        return Complex<T>{T::Min(aLeft.real, aRight.real), T::Min(aLeft.imag, aRight.imag)};
    }

    /// <summary>
    /// Complex minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vmins2(aLeft, aRight));
    }

    /// <summary>
    /// Complex minimum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Min(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hmin2(aLeft, aRight));
    }

    /// <summary>
    /// Complex maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex<T> Max(const Complex<T> &aRight) const
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Complex<T>{T(max(real, aRight.real)), T(max(imag, aRight.imag))};
    }

    /// <summary>
    /// Complex maximum
    /// </summary>
    [[nodiscard]] Complex<T> Max(const Complex<T> &aRight) const
        requires HostCode<T> && NativeNumber<T>
    {
        return Complex<T>{std::max(real, aRight.real), std::max(imag, aRight.imag)};
    }

    /// <summary>
    /// Complex maximum
    /// </summary>
    DEVICE_CODE void Max(const Complex<T> &aRight)
        requires NonNativeNumber<T>
    {
        real.Max(aRight.real);
        imag.Max(aRight.imag);
    }

    /// <summary>
    /// Complex maximum (SIMD)
    /// </summary>
    DEVICE_CODE void Max(const Complex<T> &aOther)
        requires IsShort<T> && CUDA_ONLY<T> && NativeNumber<T> && EnableSIMD<T>
    {
        *this = FromUint(__vmaxs2(*this, aOther));
    }

    /// <summary>
    /// Complex maximum (SIMD)
    /// </summary>
    DEVICE_CODE void Max(const Complex<T> &aOther)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(__hmax2(*this, aOther));
    }

    /// <summary>
    /// Complex maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Max(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Complex<T>{T(max(aLeft.real, aRight.real)), T(max(aLeft.imag, aRight.imag))};
    }

    /// <summary>
    /// Complex maximum
    /// </summary>
    [[nodiscard]] static Complex<T> Max(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        return Complex<T>{std::max(aLeft.real, aRight.real), std::max(aLeft.imag, aRight.imag)};
    }

    /// <summary>
    /// Complex maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Max(const Complex<T> &aLeft, const Complex<T> &aRight)
        requires NonNativeNumber<T>
    {
        return Complex<T>{T::Max(aLeft.real, aRight.real), T::Max(aLeft.imag, aRight.imag)};
    }

    /// <summary>
    /// Complex maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires IsShort<T> && NativeNumber<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromUint(__vmaxs2(aLeft, aRight));
    }

    /// <summary>
    /// Complex maximum (SIMD)
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector2<T> Max(const Vector2<T> &aLeft, const Vector2<T> &aRight)
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        return FromNV16BitFloat(__hmax2(aLeft, aRight));
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Round(const Complex<T> &aValue)
        requires RealFloatingPoint<T>
    {
        Complex<T> ret = aValue;
        ret.Round();
        return ret;
    }

    /// <summary>
    /// Element wise round() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<int> RoundI(const Complex<T> &aValue)
        requires RealFloatingPoint<T>
    {
        Complex<T> ret = aValue;
        ret.Round();
        return Complex<int>(ret);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE void Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        real = round(real);
        imag = round(imag);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    void Round()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        real = std::round(real);
        imag = std::round(imag);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_ONLY_CODE void Round()
        requires NonNativeNumber<T>
    {
        real.Round();
        imag.Round();
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Floor(const Complex<T> &aValue)
        requires RealFloatingPoint<T>
    {
        Complex<T> ret = aValue;
        ret.Floor();
        return ret;
    }

    /// <summary>
    /// Element wise floor() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<int> FloorI(const Complex<T> &aValue)
        requires RealFloatingPoint<T>
    {
        Complex<T> ret = aValue;
        ret.Floor();
        return Complex<int>(ret);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE void Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        real = floor(real);
        imag = floor(imag);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    void Floor()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        real = std::floor(real);
        imag = std::floor(imag);
    }

    /// <summary>
    /// Element wise floor() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE void Floor()
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2floor(*this));
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE void Floor()
        requires NonNativeNumber<T>
    {
        real.Floor();
        imag.Floor();
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Ceil(const Complex<T> &aValue)
        requires RealFloatingPoint<T>
    {
        Complex<T> ret = aValue;
        ret.Ceil();
        return ret;
    }

    /// <summary>
    /// Element wise ceil() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<int> CeilI(const Complex<T> &aValue)
        requires RealFloatingPoint<T>
    {
        Complex<T> ret = aValue;
        ret.Ceil();
        return Complex<int>(ret);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE void Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        real = ceil(real);
        imag = ceil(imag);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    void Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        real = std::ceil(real);
        imag = std::ceil(imag);
    }

    /// <summary>
    /// Element wise ceil() (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE void Ceil()
        requires Is16BitFloat<T> && CUDA_ONLY<T>
    {
        *this = FromNV16BitFloat(h2ceil(*this));
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE void Ceil()
        requires NonNativeNumber<T>
    {
        real.Ceil();
        imag.Ceil();
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> RoundNearest(const Complex<T> &aValue)
        requires RealFloatingPoint<T>
    {
        Complex<T> ret = aValue;
        ret.RoundNearest();
        return ret;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE void RoundNearest()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        real = __float2int_rn(real);
        imag = __float2int_rn(imag);
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    void RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        real = std::nearbyint(real);
        imag = std::nearbyint(imag);
    }

    /// <summary>
    /// Element wise round nearest ties to even (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE void RoundNearest()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(h2rint(*this));
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE void RoundNearest()
        requires NonNativeNumber<T>
    {
        real.RoundNearest();
        imag.RoundNearest();
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> RoundZero(const Complex<T> &aValue)
        requires RealFloatingPoint<T>
    {
        Complex<T> ret = aValue;
        ret.RoundZero();
        return ret;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE void RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        real = __float2int_rz(real);
        imag = __float2int_rz(imag);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    void RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        real = std::trunc(real);
        imag = std::trunc(imag);
    }

    /// <summary>
    /// Element wise round toward zero (SIMD)
    /// </summary>
    DEVICE_ONLY_CODE void RoundZero()
        requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
    {
        *this = FromNV16BitFloat(h2trunc(*this));
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE void RoundZero()
        requires NonNativeNumber<T>
    {
        real.RoundZero();
        imag.RoundZero();
    }
};

template <typename T, typename T2>
DEVICE_CODE Complex<T> operator+(const Complex<T> &aLeft, T2 aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{T(aLeft.real + aRight), T(aLeft.imag)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator+(T2 aLeft, const Complex<T> &aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{T(aLeft + aRight.real), T(aRight.imag)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator-(const Complex<T> &aLeft, T2 aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{T(aLeft.real - aRight), T(aLeft.imag)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator-(T2 aLeft, const Complex<T> &aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{T(aLeft - aRight.real), T(aRight.imag)};
}

template <typename T, typename T2>
DEVICE_CODE Complex<T> operator*(const Complex<T> &aLeft, T2 aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{T(aLeft.real * aRight), T(aLeft.imag * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator*(T2 aLeft, const Complex<T> &aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{T(aLeft * aRight.real), T(aLeft * aRight.imag)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator/(const Complex<T> &aLeft, T2 aRight)
    requires RealSignedNumber<T2>
{
    return Complex<T>{T(aLeft.real / aRight), T(aLeft.imag / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator/(T2 aLeft, const Complex<T> &aRight)
    requires RealSignedNumber<T2>
{
    Complex<T> ret(aLeft);
    return ret / aRight; // complex division
}

template <HostCode T2> std::ostream &operator<<(std::ostream &aOs, const Complex<T2> &aVec)
{
    aOs << aVec.real << " + " << aVec.imag << 'i';
    return aOs;
}

template <HostCode T2> std::wostream &operator<<(std::wostream &aOs, const Complex<T2> &aVec)
{
    aOs << aVec.real << " + " << aVec.imag << 'i';
    return aOs;
}

template <HostCode T2> std::istream &operator>>(std::istream &aIs, Complex<T2> &aVec)
{
    aIs >> aVec.real >> aVec.imag;
    return aIs;
}

template <HostCode T2> std::wistream &operator>>(std::wistream &aIs, Complex<T2> &aVec)
{
    aIs >> aVec.real >> aVec.imag;
    return aIs;
}
} // namespace opp
