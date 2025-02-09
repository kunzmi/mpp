#pragma once
#include "complex.h"
#include "defines.h"
#include "exception.h"
#include "needSaturationClamp.h"
#include "safeCast.h"
#include "staticCast.h"
#include <cmath>
#include <common/bfloat16.h>
#include <common/half_fp16.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <common/vector2_impl.h>
#include <complex>
#include <concepts>
#include <iostream>
#include <type_traits>

#ifdef IS_CUDA_COMPILER
#include "opp_defs.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#else
// no arguments to these intrinsics directly depend on a template parameter,
// so a declaration must be available:
opp::float2 __half22float2(opp::half2);               // NOLINT
opp::half2 __float22half2_rn(opp::float2);            // NOLINT
opp::float2 __bfloat1622float2(opp::nv_bfloat162);    // NOLINT
opp::nv_bfloat162 __float22bfloat162_rn(opp::float2); // NOLINT
#endif

namespace opp
{

#pragma region Constructors

/// <summary>
/// Initializes complex number with only real part, imag = 0
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T>::Complex(T aVal) noexcept : real(aVal), imag(static_cast<T>(0))
{
}

/// <summary>
/// Initializes complex number with only real part, imag = 0 (avoid confusion with the array constructor)
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T>::Complex(int aVal) noexcept
    requires(!IsInt<T>)
    : real(static_cast<T>(aVal)), imag(static_cast<T>(0))
{
}

/// <summary>
/// Initializes complex number with real = aVal[0], imag = aVal[1]
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T>::Complex(T aVal[2]) noexcept : real(aVal[0]), imag(aVal[1])
{
}

/// <summary>
/// Initializes complex number with real = aReal, imag = aImag
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T>::Complex(T aReal, T aImag) noexcept // NOLINT
    : real(aReal), imag(aImag)
{
}

/// <summary>
/// Convert from Vector2 of same type
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T>::Complex(const Vector2<T> &aVec) noexcept : real(aVec.x), imag(aVec.y)
{
}

/// <summary>
/// Convert from std::complex of same type
/// </summary>
template <RealSignedNumber T>
Complex<T>::Complex(const std::complex<T> &aCplx) noexcept : real(aCplx.real()), imag(aCplx.imag())
{
}

/// <summary>
/// Type conversion with saturation if needed<para/>
/// E.g.: when converting int to byte, values are clamped to 0..255<para/>
/// But when converting byte to int, no clamping operation is performed.
/// </summary>
template <RealSignedNumber T>
template <RealSignedNumber T2>
DEVICE_CODE Complex<T>::Complex(const Complex<T2> &aCplx) noexcept
    // Disable the non-const variant for half and bfloat to / from float,
    // otherwise the const specialization will never be picked up:
    requires(!(IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsBFloat16<T2> && CUDA_ONLY<T>) &&
             !(IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>) && !(IsFloat<T> && IsHalfFp16<T2> && CUDA_ONLY<T>))
{
    if constexpr (need_saturation_clamp_v<T2, T>)
    {
        Complex<T2> temp(aCplx);
        temp.template ClampToTargetType<T>();
        real = StaticCast<T2, T>(temp.real);
        imag = StaticCast<T2, T>(temp.imag);
    }
    else
    {
        real = StaticCast<T2, T>(aCplx.real);
        imag = StaticCast<T2, T>(aCplx.imag);
    }
}

/// <summary>
/// Type conversion with saturation if needed<para/>
/// E.g.: when converting int to byte, values are clamped to 0..255<para/>
/// But when converting byte to int, no clamping operation is performed.<para/>
/// If we can modify the input variable, no need to allocate temporary storage for clamping.
/// </summary>
template <RealSignedNumber T>
template <RealSignedNumber T2>
DEVICE_CODE Complex<T>::Complex(Complex<T2> &aVec) noexcept
    requires(!std::same_as<T, T2>)
{
    if constexpr (need_saturation_clamp_v<T2, T>)
    {
        aVec.template ClampToTargetType<T>();
    }
    real = StaticCast<T2, T>(aVec.real);
    imag = StaticCast<T2, T>(aVec.imag);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to BFloat2
/// </summary>
template <RealSignedNumber T>
template <Number T2>
DEVICE_CODE Complex<T>::Complex(const Complex<T2> &aVec) noexcept
    requires IsBFloat16<T> && IsFloat<T2> && CUDA_ONLY<T>
{
    const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
    nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
    *thisPtr              = __float22bfloat162_rn(*aVecPtr);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to BFloat2
/// </summary>
template <RealSignedNumber T>
template <Number T2>
DEVICE_CODE Complex<T>::Complex(const Complex<T2> &aVec, RoundingMode aRoundingMode) noexcept
    requires IsBFloat16<T> && IsFloat<T2>
{
    if constexpr (CUDA_ONLY<T>)
    {
        if (aRoundingMode == RoundingMode::NearestTiesToEven)
        {
            const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
            nv_bfloat162 *thisPtr = reinterpret_cast<nv_bfloat162 *>(this);
            *thisPtr              = __float22bfloat162_rn(*aVecPtr);
        }
        else
        {
            real = BFloat16(aVec.real, aRoundingMode);
            imag = BFloat16(aVec.imag, aRoundingMode);
        }
    }
    else
    {
        real = BFloat16(aVec.real, aRoundingMode);
        imag = BFloat16(aVec.imag, aRoundingMode);
    }
}

/// <summary>
/// Type conversion using CUDA intrinsics for BFloat2 to float2
/// </summary>
template <RealSignedNumber T>
template <IsBFloat16 T2>
DEVICE_CODE Complex<T>::Complex(const Complex<T2> &aVec) noexcept
    requires IsFloat<T> && CUDA_ONLY<T>
{
    const nv_bfloat162 *aVecPtr = reinterpret_cast<const nv_bfloat162 *>(&aVec);
    float2 *thisPtr             = reinterpret_cast<float2 *>(this);
    *thisPtr                    = __bfloat1622float2(*aVecPtr);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float2 to half2
/// </summary>
template <RealSignedNumber T>
template <Number T2>
DEVICE_CODE Complex<T>::Complex(const Complex<T2> &aVec) noexcept
    requires IsHalfFp16<T> && IsFloat<T2> && CUDA_ONLY<T>
{
    const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
    half2 *thisPtr        = reinterpret_cast<half2 *>(this);
    *thisPtr              = __float22half2_rn(*aVecPtr);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float to half
/// </summary>
template <RealSignedNumber T>
template <Number T2>
DEVICE_CODE Complex<T>::Complex(const Complex<T2> &aVec, RoundingMode aRoundingMode) noexcept
    requires IsHalfFp16<T> && IsFloat<T2>
{
    if constexpr (CUDA_ONLY<T>)
    {
        if (aRoundingMode == RoundingMode::NearestTiesToEven)
        {
            const float2 *aVecPtr = reinterpret_cast<const float2 *>(&aVec);
            half2 *thisPtr        = reinterpret_cast<half2 *>(this);
            *thisPtr              = __float22half2_rn(*aVecPtr);
        }
        else
        {
            real = HalfFp16(aVec.real, aRoundingMode);
            imag = HalfFp16(aVec.imag, aRoundingMode);
        }
    }
    else
    {
        real = HalfFp16(aVec.real, aRoundingMode);
        imag = HalfFp16(aVec.imag, aRoundingMode);
    }
}

/// <summary>
/// Type conversion using CUDA intrinsics for half2 to float2
/// </summary>
template <RealSignedNumber T>
template <IsHalfFp16 T2>
DEVICE_CODE Complex<T>::Complex(const Complex<T2> &aVec) noexcept
    requires IsFloat<T> && CUDA_ONLY<T>
{
    const half2 *aVecPtr = reinterpret_cast<const half2 *>(&aVec);
    float2 *thisPtr      = reinterpret_cast<float2 *>(this);
    *thisPtr             = __half22float2(*aVecPtr);
}

/// <summary>
/// Usefull constructor for SIMD instructions
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::FromUint(const uint &aUint) noexcept
    requires TwoBytesSizeType<T>
{
    return Complex(*reinterpret_cast<const Complex<T> *>(&aUint));
}

/// <summary>
/// Usefull constructor for SIMD instructions
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::FromNV16BitFloat(const nv_bfloat162 &aNVBfloat2) noexcept
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    return Complex(*reinterpret_cast<const Complex<T> *>(&aNVBfloat2));
}

/// <summary>
/// Usefull constructor for SIMD instructions
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::FromNV16BitFloat(const half2 &aNVHalf2) noexcept
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    return Complex(*reinterpret_cast<const Complex<T> *>(&aNVHalf2));
}

// if we make those converters public we will get in trouble with some T constructors / operators
/// <summary>
/// converter to uint for SIMD operations
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T>::operator const uint &() const
    requires TwoBytesSizeType<T>
{
    return *reinterpret_cast<const uint *>(this);
}

/// <summary>
/// converter to uint for SIMD operations
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T>::operator uint &()
    requires TwoBytesSizeType<T>
{
    return *reinterpret_cast<uint *>(this);
}

/// <summary>
/// converter to nv_bfloat162 for SIMD operations
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T>::operator const nv_bfloat162 &() const
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    return *reinterpret_cast<const nv_bfloat162 *>(this);
}

/// <summary>
/// converter to nv_bfloat162 for SIMD operations
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T>::operator nv_bfloat162 &()
    requires IsBFloat16<T> && CUDA_ONLY<T>
{
    return *reinterpret_cast<nv_bfloat162 *>(this);
}

/// <summary>
/// converter to half2 for SIMD operations
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T>::operator const half2 &() const
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    return *reinterpret_cast<const half2 *>(this);
}

/// <summary>
/// converter to half2 for SIMD operations
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T>::operator half2 &()
    requires IsHalfFp16<T> && CUDA_ONLY<T>
{
    return *reinterpret_cast<half2 *>(this);
}
#pragma endregion

#pragma region Operators
// No operators for < or > as complex numbers have no ordering

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <RealSignedNumber T>
bool Complex<T>::EqEps(Complex aLeft, Complex aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && HostCode<T>
{
    MakeNANandINFValid(aLeft.real, aRight.real);
    MakeNANandINFValid(aLeft.imag, aRight.imag);

    bool res = std::abs(aLeft.real - aRight.real) <= aEpsilon;
    res &= std::abs(aLeft.imag - aRight.imag) <= aEpsilon;
    return res;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE bool Complex<T>::EqEps(Complex aLeft, Complex aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && DeviceCode<T>
{
    MakeNANandINFValid(aLeft.real, aRight.real);
    MakeNANandINFValid(aLeft.imag, aRight.imag);

    bool res = abs(aLeft.real - aRight.real) <= aEpsilon;
    res &= abs(aLeft.imag - aRight.imag) <= aEpsilon;
    return res;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE bool Complex<T>::EqEps(Complex aLeft, Complex aRight, T aEpsilon)
    requires Is16BitFloat<T>
{
    MakeNANandINFValid(aLeft.real, aRight.real);
    MakeNANandINFValid(aLeft.imag, aRight.imag);

    bool res = T::Abs(aLeft.real - aRight.real) <= aEpsilon;
    res &= T::Abs(aLeft.imag - aRight.imag) <= aEpsilon;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <RealSignedNumber T> DEVICE_CODE bool Complex<T>::operator==(const Complex &aOther) const
{
    bool res = real == aOther.real;
    res &= imag == aOther.imag;
    return res;
}

/// <summary>
/// Returns true if any element comparison is true
/// </summary>
template <RealSignedNumber T> DEVICE_CODE bool Complex<T>::operator!=(const Complex &aOther) const
{
    bool res = real != aOther.real;
    res |= imag != aOther.imag;
    return res;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE bool Complex<T>::operator==(const Complex &aOther) const
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpeq2(*this, aOther) == 0xFFFFFFFFU; // NOLINT
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE bool Complex<T>::operator!=(const Complex &aOther) const
    requires IsShort<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __vcmpne2(*this, aOther) != 0U;
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE bool Complex<T>::operator==(const Complex &aOther) const
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __hbeq2(*this, aOther);
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE bool Complex<T>::operator!=(const Complex &aOther) const
    requires IsBFloat16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    // __hbne2 returns true only if both elements are != but we need true if any element is !=
    // so we use hbeq and negate the result
    return !(__hbeq2(*this, aOther));
}

/// <summary>
/// Returns true if each element comparison is true (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE bool Complex<T>::operator==(const Complex &aOther) const
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    return __hbeq2(*this, aOther);
}

/// <summary>
/// Returns true if any element comparison is true (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE bool Complex<T>::operator!=(const Complex &aOther) const
    requires IsHalfFp16<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    // __hbne2 returns true only if both elements are != but we need true if any element is !=
    // so we use hbeq and negate the result
    return !(__hbeq2(*this, aOther));
}

/// <summary>
/// Negation
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> Complex<T>::operator-() const
{
    return Complex<T>(static_cast<T>(-real), static_cast<T>(-imag));
}

/// <summary>
/// Negation (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::operator-() const
    requires IsShort<T> && CUDA_ONLY<T>
{
    return FromUint(__vnegss2(*this));
}

/// <summary>
/// Negation (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::operator-() const
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    return FromNV16BitFloat(__hneg2(*this));
}

/// <summary>
/// Complex addition (only real part)
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::operator+=(T aOther)
{
    real += aOther;
    return *this;
}

/// <summary>
/// Complex addition
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::operator+=(const Complex &aOther)
{
    real += aOther.real;
    imag += aOther.imag;
    return *this;
}

/// <summary>
/// Complex addition SIMD
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::operator+=(const Complex &aOther)
    requires IsShort<T> && CUDA_ONLY<T>
{
    *this = FromUint(__vaddss2(*this, aOther));
    return *this;
}

/// <summary>
/// Complex addition SIMD
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::operator+=(const Complex &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    *this = FromNV16BitFloat(__hadd2(*this, aOther));
    return *this;
}

/// <summary>
/// Complex addition
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> Complex<T>::operator+(const Complex &aOther) const
{
    return Complex<T>{static_cast<T>(real + aOther.real), static_cast<T>(imag + aOther.imag)};
}

/// <summary>
/// Complex subtraction (only real part)
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::operator-=(T aOther)
{
    real -= aOther;
    return *this;
}

/// <summary>
/// Complex subtraction
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::operator-=(const Complex &aOther)
{
    real -= aOther.real;
    imag -= aOther.imag;
    return *this;
}

/// <summary>
/// Complex subtraction SIMD
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::operator-=(const Complex &aOther)
    requires IsShort<T> && CUDA_ONLY<T>
{
    *this = FromUint(__vsubss2(*this, aOther));
    return *this;
}

/// <summary>
/// Complex subtraction SIMD
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::operator-=(const Complex &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    *this = FromNV16BitFloat(__hsub2(*this, aOther));
    return *this;
}

/// <summary>
/// Complex subtraction (inverted inplace sub: this = aOther - this)
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::SubInv(const Complex &aOther)
{
    real = aOther.real - real;
    imag = aOther.imag - imag;
    return *this;
}

/// <summary>
/// Complex subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::SubInv(const Complex &aOther)
    requires IsShort<T> && CUDA_ONLY<T>
{
    *this = FromUint(__vsubss2(aOther, *this));
    return *this;
}

/// <summary>
/// Complex subtraction SIMD (inverted inplace sub: this = aOther - this)
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::SubInv(const Complex &aOther)
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    *this = FromNV16BitFloat(__hsub2(aOther, *this));
    return *this;
}

/// <summary>
/// Complex subtraction
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> Complex<T>::operator-(const Complex &aOther) const
{
    return Complex<T>{static_cast<T>(real - aOther.real), static_cast<T>(imag - aOther.imag)};
}

/// <summary>
/// Complex subtraction SIMD
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::operator-(const Complex &aOther) const
    requires IsShort<T> && CUDA_ONLY<T>
{
    return FromUint(__vsubss2(*this, aOther));
}

/// <summary>
/// Complex subtraction SIMD
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::operator-(const Complex &aOther) const
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    return FromNV16BitFloat(__hsub2(*this, aOther));
}

/// <summary>
/// Complex multiplication with real number
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::operator*=(T aOther)
{
    real *= aOther;
    imag *= aOther;
    return *this;
}

/// <summary>
/// Complex multiplication
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::operator*=(const Complex &aOther)
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
template <RealSignedNumber T> DEVICE_CODE Complex<T> Complex<T>::operator*(const Complex &aOther) const
{
    const T tempReal = real * aOther.real - imag * aOther.imag;
    const T tempImag = real * aOther.imag + imag * aOther.real;
    return Complex<T>(tempReal, tempImag);
}

/// <summary>
/// Complex division with real number
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::operator/=(T aOther)
{
    real /= aOther;
    imag /= aOther;
    return *this;
}

/// <summary>
/// Complex division
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::operator/=(const Complex &aOther)
{
    T denom = aOther.real * aOther.real + aOther.imag * aOther.imag;
    if constexpr (RealIntegral<T>)
    { // floats will denormalize to inf, but likely just an overflow
        if (denom == T(0))
        {
            real = 0;
            imag = 0;
            return *this;
        }
    }
    T tempReal = real * aOther.real + imag * aOther.imag;
    T tempImag = imag * aOther.real - real * aOther.imag;
    real       = tempReal / denom;
    imag       = tempImag / denom;
    return *this;
}

/// <summary>
/// Complex division (inverted inplace div: this = aOther / this)
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::DivInv(const Complex &aOther)
{
    T denom = real * real + imag * imag;
    if constexpr (RealIntegral<T>)
    { // floats will denormalize to inf, but likely just an overflow
        if (denom == T(0))
        {
            real = 0;
            imag = 0;
            return *this;
        }
    }
    T tempReal = aOther.real * real + aOther.imag * imag;
    T tempImag = aOther.imag * real - aOther.real * imag;
    real       = tempReal / denom;
    imag       = tempImag / denom;
    return *this;
}

/// <summary>
/// Complex division
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> Complex<T>::operator/(const Complex &aOther) const
{
    T denom = aOther.real * aOther.real + aOther.imag * aOther.imag;
    if constexpr (RealIntegral<T>)
    { // floats will denormalize to inf, but likely just an overflow
        if (denom == T(0))
        {
            return {static_cast<T>(0), static_cast<T>(0)};
        }
    }
    T tempReal = real * aOther.real + imag * aOther.imag;
    T tempImag = imag * aOther.real - real * aOther.imag;
    return {static_cast<T>(tempReal / denom), static_cast<T>(tempImag / denom)};
}
#pragma endregion

#pragma region Methods
#pragma region Exp
/// <summary>
/// Complex exponential
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::Exp()
    requires HostCode<T> && NativeNumber<T>
{
    const T e_real   = std::exp(real);
    const T cos_imag = std::cos(imag);
    const T sin_imag = std::sin(imag);
    real             = e_real * cos_imag;
    imag             = e_real * sin_imag;
    return *this;
}

/// <summary>
/// Complex exponential
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::Exp()
    requires HostCode<T> && Is16BitFloat<T>
{
    const T e_real   = T(std::exp(static_cast<float>(real)));
    const T cos_imag = T(std::cos(static_cast<float>(imag)));
    const T sin_imag = T(std::sin(static_cast<float>(imag)));
    real             = static_cast<T>(e_real * cos_imag);
    imag             = static_cast<T>(e_real * sin_imag);
    return *this;
}

/// <summary>
/// Complex exponential
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Exp()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    const T e_real = exp(real);
    T cos_imag;
    T sin_imag;

    sincos(imag, &sin_imag, &cos_imag);
    real = e_real * cos_imag;
    imag = e_real * sin_imag;
    return *this;
}
/// <summary>
/// Complex exponential
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Exp()
    requires DeviceCode<T> && Is16BitFloat<T>
{
    const T e_real   = T::Exp(real);
    const T cos_imag = T::Cos(imag);
    const T sin_imag = T::Sin(imag);
    real             = e_real * cos_imag;
    imag             = e_real * sin_imag;
    return *this;
}

/// <summary>
/// Complex exponential
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> Complex<T>::Exp(const Complex &aVec)
{
    Complex ret = aVec;
    ret.Exp();
    return ret;
}
#pragma endregion

#pragma region Log
/// <summary>
/// Complex natural logarithm
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::Ln()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    const T log_real = std::log(Magnitude()); // don't modify the real value before computing angle
    imag             = Angle();
    real             = log_real;
    return *this;
}

/// <summary>
/// Complex natural logarithm
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Ln()
    requires NonNativeNumber<T>
{
    const T log_real = T::Ln(Magnitude()); // don't modify the real value before computing angle
    imag             = Angle();
    real             = log_real;
    return *this;
}

/// <summary>
/// Complex natural logarithm
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Ln()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    const T log_real = log(Magnitude()); // don't modify the real value before computing angle
    imag             = Angle();
    real             = log_real;
    return *this;
}

/// <summary>
/// Complex natural logarithm
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::Ln(const Complex &aVec)
    requires RealFloatingPoint<T>
{
    Complex ret = aVec;
    ret.Ln();
    return ret;
}
#pragma endregion

#pragma region Sqr
/// <summary>
/// Complex square
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::Sqr()
{
    *this = *this * *this;
    return *this;
}

/// <summary>
/// Complex square
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> Complex<T>::Sqr(const Complex &aVec)
{
    return aVec * aVec;
}
#pragma endregion

#pragma region Sqrt
/// <summary>
/// Complex square root
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::Sqrt()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    T mag            = Magnitude();
    const T sqr_real = std::sqrt((mag + real) * static_cast<T>(0.5));
    const T sqr_imag = GetSign(imag) * std::sqrt((mag - real) * static_cast<T>(0.5));

    real = sqr_real;
    imag = sqr_imag;
    return *this;
}

/// <summary>
/// Complex square root
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Sqrt()
    requires NonNativeNumber<T>
{
    T mag            = Magnitude();
    const T sqr_real = T::Sqrt((mag + real) * static_cast<T>(0.5));
    const T sqr_imag = imag.GetSign() * T::Sqrt((mag - real) * static_cast<T>(0.5));

    real = sqr_real;
    imag = sqr_imag;
    return *this;
}

/// <summary>
/// Complex square root
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Sqrt()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    T mag            = Magnitude();
    const T sqr_real = sqrt((mag + real) * static_cast<T>(0.5));
    const T sqr_imag = GetSign(imag) * sqrt((mag - real) * static_cast<T>(0.5));

    real = sqr_real;
    imag = sqr_imag;
    return *this;
}

/// <summary>
/// Complex square root
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::Sqrt(const Complex &aVec)
    requires RealFloatingPoint<T>
{
    Complex ret = aVec;
    ret.Sqrt();
    return ret;
}
#pragma endregion

/// <summary>
/// Conjugate complex
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::Conj()
{
    imag = -imag;
    return *this;
}

/// <summary>
/// Conjugate complex
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> Complex<T>::Conj(const Complex<T> &aValue)
{
    return {aValue.real, static_cast<T>(-aValue.imag)};
}

/// <summary>
/// Conjugate complex multiplication: this * conj(aOther)
/// </summary>
template <RealSignedNumber T> DEVICE_CODE Complex<T> &Complex<T>::ConjMul(const Complex<T> &aOther)
{
    const T realTemp = (real * aOther.real) + (imag * aOther.imag);
    const T imagTemp = (aOther.real * imag) - (aOther.imag * real);
    real             = realTemp;
    imag             = imagTemp;
    return *this;
}

/// <summary>
/// Conjugate complex multiplication: aLeft * conj(aRight)
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::ConjMul(const Complex<T> &aLeft, const Complex<T> &aRight)
{
    const T realTemp = (aLeft.real * aRight.real) + (aLeft.imag * aRight.imag);
    const T imagTemp = (aRight.real * aLeft.imag) - (aRight.imag * aLeft.real);

    return {realTemp, imagTemp};
}

/// <summary>
/// Complex magnitude |a+bi|
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE T Complex<T>::Magnitude() const
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    return sqrt(MagnitudeSqr());
}

/// <summary>
/// Complex magnitude |a+bi|
/// </summary>
template <RealSignedNumber T>
T Complex<T>::Magnitude() const
    requires HostCode<T> && NativeFloatingPoint<T>
{
    return std::sqrt(MagnitudeSqr());
}

/// <summary>
/// Complex magnitude |a+bi|
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE T Complex<T>::Magnitude() const
    requires NonNativeNumber<T>
{
    return T::Sqrt(MagnitudeSqr());
}

/// <summary>
/// Complex magnitude squared |a+bi|^2
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE T Complex<T>::MagnitudeSqr() const
    requires RealFloatingPoint<T>
{
    return real * real + imag * imag;
}

/// <summary>
/// Angle between real and imaginary of a complex number (atan2(image, real))
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE T Complex<T>::Angle() const
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    return atan2(imag, real);
}

/// <summary>
/// Angle between real and imaginary of a complex number (atan2(image, real))
/// </summary>
template <RealSignedNumber T>
T Complex<T>::Angle() const
    requires HostCode<T> && NativeFloatingPoint<T>
{
    return std::atan2(imag, real);
}

/// <summary>
/// Angle between real and imaginary of a complex number (atan2(image, real))
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE T Complex<T>::Angle() const
    requires NonNativeNumber<T>
{
    // cast bfloat16 and hfloat16 to float for atan2:
    return static_cast<T>(atan2f(static_cast<float>(imag), static_cast<float>(real)));
}

/// <summary>
/// Complex clamp to value range
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Clamp(T aMinVal, T aMaxVal)
    requires DeviceCode<T> && NativeNumber<T>
{
    real = max(aMinVal, min(real, aMaxVal));
    imag = max(aMinVal, min(imag, aMaxVal));
    return *this;
}

/// <summary>
/// Complex clamp to value range
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::Clamp(T aMinVal, T aMaxVal)
    requires HostCode<T> && NativeNumber<T>
{
    real = std::max(aMinVal, std::min(real, aMaxVal));
    imag = std::max(aMinVal, std::min(imag, aMaxVal));
    return *this;
}

/// <summary>
/// Complex clamp to value range
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Clamp(T aMinVal, T aMaxVal)
    requires NonNativeNumber<T>
{
    real = T::Max(aMinVal, T::Min(real, aMaxVal));
    imag = T::Max(aMinVal, T::Min(imag, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to maximum value range of given target type
/// </summary>
template <RealSignedNumber T>
template <Number TTarget>
DEVICE_CODE Complex<T> &Complex<T>::ClampToTargetType() noexcept
    requires(need_saturation_clamp_v<T, TTarget>)
{
    return Clamp(numeric_limits_conversion<T, TTarget>::lowest(), numeric_limits_conversion<T, TTarget>::max());
}

/// <summary>
/// Component wise clamp to maximum value range of given target type<para/>
/// NOP in case no saturation clamping is needed.
/// </summary>
template <RealSignedNumber T>
template <Number TTarget>
DEVICE_CODE Complex<T> &Complex<T>::ClampToTargetType() noexcept
    requires(!need_saturation_clamp_v<T, TTarget>)
{
    return *this;
}

/// <summary>
/// Element wise round()
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::Round(const Complex<T> &aValue)
    requires RealFloatingPoint<T>
{
    Complex<T> ret = aValue;
    ret.Round();
    return ret;
}

/// <summary>
/// Element wise round()
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Round()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    real = round(real);
    imag = round(imag);
    return *this;
}

/// <summary>
/// Element wise round()
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::Round()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    real = std::round(real);
    imag = std::round(imag);
    return *this;
}

/// <summary>
/// Element wise round()
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE Complex<T> &Complex<T>::Round()
    requires NonNativeNumber<T>
{
    real.Round();
    imag.Round();
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::Floor(const Complex<T> &aValue)
    requires RealFloatingPoint<T>
{
    Complex<T> ret = aValue;
    ret.Floor();
    return ret;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Floor()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    real = floor(real);
    imag = floor(imag);
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::Floor()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    real = std::floor(real);
    imag = std::floor(imag);
    return *this;
}

/// <summary>
/// Element wise floor() (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE Complex<T> &Complex<T>::Floor()
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    *this = FromNV16BitFloat(h2floor(*this));
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE Complex<T> &Complex<T>::Floor()
    requires NonNativeNumber<T>
{
    real.Floor();
    imag.Floor();
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::Ceil(const Complex<T> &aValue)
    requires RealFloatingPoint<T>
{
    Complex<T> ret = aValue;
    ret.Ceil();
    return ret;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::Ceil()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    real = ceil(real);
    imag = ceil(imag);
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::Ceil()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    real = std::ceil(real);
    imag = std::ceil(imag);
    return *this;
}

/// <summary>
/// Element wise ceil() (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE Complex<T> &Complex<T>::Ceil()
    requires Is16BitFloat<T> && CUDA_ONLY<T>
{
    *this = FromNV16BitFloat(h2ceil(*this));
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE Complex<T> &Complex<T>::Ceil()
    requires NonNativeNumber<T>
{
    real.Ceil();
    imag.Ceil();
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::RoundNearest(const Complex<T> &aValue)
    requires RealFloatingPoint<T>
{
    Complex<T> ret = aValue;
    ret.RoundNearest();
    return ret;
}

/// <summary>
/// Element wise round nearest ties to even
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::RoundNearest()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    real = rint(real);
    imag = rint(imag);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even<para/>
/// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::RoundNearest()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    real = std::nearbyint(real);
    imag = std::nearbyint(imag);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE Complex<T> &Complex<T>::RoundNearest()
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(h2rint(*this));
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE Complex<T> &Complex<T>::RoundNearest()
    requires NonNativeNumber<T>
{
    real.RoundNearest();
    imag.RoundNearest();
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> Complex<T>::RoundZero(const Complex<T> &aValue)
    requires RealFloatingPoint<T>
{
    Complex<T> ret = aValue;
    ret.RoundZero();
    return ret;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <RealSignedNumber T>
DEVICE_CODE Complex<T> &Complex<T>::RoundZero()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    real = trunc(real);
    imag = trunc(imag);
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <RealSignedNumber T>
Complex<T> &Complex<T>::RoundZero()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    real = std::trunc(real);
    imag = std::trunc(imag);
    return *this;
}

/// <summary>
/// Element wise round toward zero (SIMD)
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE Complex<T> &Complex<T>::RoundZero()
    requires Is16BitFloat<T> && CUDA_ONLY<T> && EnableSIMD<T>
{
    *this = FromNV16BitFloat(h2trunc(*this));
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <RealSignedNumber T>
DEVICE_ONLY_CODE Complex<T> &Complex<T>::RoundZero()
    requires NonNativeNumber<T>
{
    real.RoundZero();
    imag.RoundZero();
    return *this;
}

template <opp::HostCode T2> std::ostream &operator<<(std::ostream &aOs, const opp::Complex<T2> &aVec)
{
    if constexpr (opp::ByteSizeType<T2>)
    {
        const int real = static_cast<int>(aVec.real); // NOLINT
        const int imag = static_cast<int>(aVec.imag); // NOLINT
        aOs << '(' << real;

        if (imag < 0)
        {
            aOs << " - " << std::abs(imag) << 'i';
        }
        else if (imag > 0)
        {
            aOs << " + " << imag << 'i';
        }
        aOs << ')';
    }
    else
    {
        aOs << '(' << aVec.real;

        if (aVec.imag < 0)
        {
            aOs << " - " << std::abs(aVec.imag) << 'i';
        }
        else if (aVec.imag > 0)
        {
            aOs << " + " << aVec.imag << 'i';
        }
        aOs << ')';
    }

    return aOs;
}

template <opp::HostCode T2> std::wostream &operator<<(std::wostream &aOs, const opp::Complex<T2> &aVec)
{
    if constexpr (opp::ByteSizeType<T2>)
    {
        const int real = static_cast<int>(aVec.real); // NOLINT
        const int imag = static_cast<int>(aVec.imag); // NOLINT
        aOs << '(' << real;

        if (imag < 0)
        {
            aOs << " - " << std::abs(imag) << 'i';
        }
        else if (imag > 0)
        {
            aOs << " + " << imag << 'i';
        }
        aOs << ')';
    }
    else
    {
        aOs << '(' << aVec.real;

        if (aVec.imag < 0)
        {
            aOs << " - " << std::abs(aVec.imag) << 'i';
        }
        else if (aVec.imag > 0)
        {
            aOs << " + " << aVec.imag << 'i';
        }
        aOs << ')';
    }

    return aOs;
}

template <opp::HostCode T2> std::istream &operator>>(std::istream &aIs, opp::Complex<T2> &aVec)
{
    if constexpr (opp::ByteSizeType<T2>)
    {
        int real{};
        int imag{};
        aIs >> real >> imag;
        aVec.real = static_cast<T2>(real);
        aVec.imag = static_cast<T2>(imag);
    }
    else
    {
        aIs >> aVec.real >> aVec.imag;
    }
    return aIs;
}

template <opp::HostCode T2> std::wistream &operator>>(std::wistream &aIs, opp::Complex<T2> &aVec)
{
    if constexpr (opp::ByteSizeType<T2>)
    {
        int real{};
        int imag{};
        aIs >> real >> imag;
        aVec.real = static_cast<T2>(real);
        aVec.imag = static_cast<T2>(imag);
    }
    else
    {
        aIs >> aVec.real >> aVec.imag;
    }
    return aIs;
}

} // namespace opp