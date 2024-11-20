#pragma once
#include "defines.h"
#include "exception.h"
#include "safeCast.h"
#include <cmath>
#include <concepts>
#include <iostream>
#include <type_traits>

namespace opp
{

/// <summary>
/// Our own definition of a complex number, that we can use on device and host
/// </summary>
template <SignedNumber T> struct alignas(2 * sizeof(T)) Complex
{
    T real;
    T imag;

    /// <summary>
    /// Default constructor does not initializes the members
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
    /// Type conversion
    /// </summary>
    template <SignedNumber T2> Complex(const Complex<T2> &aVec) noexcept : real(T(aVec.real)), imag(T(aVec.imag))
    {
        assert(check_is_safe_cast<T>(aVec.real));
        assert(check_is_safe_cast<T>(aVec.imag));
    }

    ~Complex() = default;

    Complex(const Complex &) noexcept            = default;
    Complex(Complex &&) noexcept                 = default;
    Complex &operator=(const Complex &) noexcept = default;
    Complex &operator=(Complex &&) noexcept      = default;

    auto operator<=>(const Complex &) const = default;

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-() const
    {
        return Complex<T>(T(-real), T(-imag));
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
    /// Complex subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator-(const Complex &aOther) const
    {
        return Complex<T>{T(real - aOther.real), T(imag - aOther.imag)};
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
    /// Complex division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex operator/(const Complex &aOther) const
    {
        T denom    = aOther.real * aOther.real + aOther.imag * aOther.imag;
        T tempReal = real * aOther.real + imag * aOther.imag;
        T tempImag = imag * aOther.real - real * aOther.imag;
        return Complex<T>{T(tempReal / denom), T(tempImag / denom)};
    }

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
    /// Vector length (L2 norm)
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires DeviceCode<T> && FloatingPoint<T>
    {
        return sqrt(real * real + imag * imag);
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] T Magnitude() const
        requires HostCode<T> && FloatingPoint<T>
    {
        return std::sqrt(real * real + imag * imag);
    }

    /// <summary>
    /// Squared vector length
    /// </summary>
    DEVICE_CODE [[nodiscard]] T MagnitudeSqr() const
        requires FloatingPoint<T>
    {
        return real * real + imag * imag;
    }

    /// <summary>
    /// Normalizes the vector components
    /// </summary>
    DEVICE_CODE void Normalize()
        requires FloatingPoint<T>
    {
        *this = *this / Magnitude();
    }

    /// <summary>
    /// Normalizes a vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Normalize(const Complex<T> &aValue)
        requires FloatingPoint<T>
    {
        Complex<T> ret = aValue;
        ret.Normalize();
        return ret;
    }

    /// <summary>
    /// Complex clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T>
    {
        real = max(aMinVal, min(real, aMaxVal));
        imag = max(aMinVal, min(imag, aMaxVal));
    }

    /// <summary>
    /// Complex clamp to value range
    /// </summary>
    void Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T>
    {
        real = std::max(aMinVal, std::min(real, aMaxVal));
        imag = std::max(aMinVal, std::min(imag, aMaxVal));
    }

    /// <summary>
    /// Complex minimum
    /// </summary>
    [[nodiscard]] Complex<T> Min(const Complex<T> &aRight) const
        requires HostCode<T>
    {
        return Complex<T>{std::min(real, aRight.real), std::min(imag, aRight.imag)};
    }

    /// <summary>
    /// Complex minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex<T> Min(const Complex<T> &aRight) const
        requires DeviceCode<T>
    {
        return Complex<T>{min(real, aRight.real), min(imag, aRight.imag)};
    }

    /// <summary>
    /// Complex maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] Complex<T> Max(const Complex<T> &aRight) const
        requires DeviceCode<T>
    {
        return Complex<T>{max(real, aRight.real), max(imag, aRight.imag)};
    }

    /// <summary>
    /// Complex maximum
    /// </summary>
    [[nodiscard]] Complex<T> Max(const Complex<T> &aRight) const
        requires HostCode<T>
    {
        return Complex<T>{std::max(real, aRight.real), std::max(imag, aRight.imag)};
    }

    /// <summary>
    /// Complex minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Min(const Complex<T> &aLeft, const Complex<T> &aRight)
    {
        return aLeft.Min(aRight);
    }

    /// <summary>
    /// Complex maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Complex<T> Max(const Complex<T> &aLeft, const Complex<T> &aRight)
    {
        return aLeft.Max(aRight);
    }
};

template <typename T, typename T2>
DEVICE_CODE Complex<T> operator+(const Complex<T> &aLeft, T2 aRight)
    requires SignedNumber<T2>
{
    return Complex<T>{T(aLeft.real + aRight), T(aLeft.imag)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator+(T2 aLeft, const Complex<T> &aRight)
    requires SignedNumber<T2>
{
    return Complex<T>{T(aLeft + aRight.real), T(aRight.imag)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator-(const Complex<T> &aLeft, T2 aRight)
    requires SignedNumber<T2>
{
    return Complex<T>{T(aLeft.real - aRight), T(aLeft.imag)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator-(T2 aLeft, const Complex<T> &aRight)
    requires SignedNumber<T2>
{
    return Complex<T>{T(aLeft - aRight.real), T(aRight.imag)};
}

template <typename T, typename T2>
DEVICE_CODE Complex<T> operator*(const Complex<T> &aLeft, T2 aRight)
    requires SignedNumber<T2>
{
    return Complex<T>{T(aLeft.real * aRight), T(aLeft.imag * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator*(T2 aLeft, const Complex<T> &aRight)
    requires SignedNumber<T2>
{
    return Complex<T>{T(aLeft * aRight.real), T(aLeft * aRight.imag)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator/(const Complex<T> &aLeft, T2 aRight)
    requires SignedNumber<T2>
{
    return Complex<T>{T(aLeft.real / aRight), T(aLeft.imag / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Complex<T> operator/(T2 aLeft, const Complex<T> &aRight)
    requires SignedNumber<T2>
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

template <typename T> struct isComplexType : std::false_type
{
};
template <typename T> struct isComplexType<Complex<T>> : std::true_type
{
};

template <typename T> struct remove_complex
{
    using type = void;
};
template <typename T> struct remove_complex<Complex<T>>
{
    using type = T;
};

template <typename T>
concept ComplexType = isComplexType<T>::value;

template <typename T>
concept IntComplexType = isComplexType<T>::value && Integral<typename remove_complex<T>::type>;

template <typename T>
concept FloatingComplexType = isComplexType<T>::value && FloatingPoint<typename remove_complex<T>::type>;
} // namespace opp
