#pragma once
#include "defines.h"
#include "exception.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "numeric_limits.h"
#include "safeCast.h"
#include "vector_typetraits.h"
#include <cmath>
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

enum class Axis1D
{
    X = 0
};

inline std::ostream &operator<<(std::ostream &aOs, const Axis1D &aAxis)
{
    switch (aAxis)
    {
        case Axis1D::X:
            aOs << 'X';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X (0).";
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const Axis1D &aAxis)
{
    switch (aAxis)
    {
        case Axis1D::X:
            aOs << 'X';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X (0).";
    return aOs;
}

/// <summary>
/// A two T component vector. Can replace CUDA's Vector1 types
/// </summary>
template <Number T> struct alignas(sizeof(T)) Vector1
{
    T x;

    template <typename T2> struct same_vector_size_different_type
    {
        using vector = Vector1<T2>;
    };

    template <typename T2>
    using same_vector_size_different_type_t = typename same_vector_size_different_type<T2>::vector;

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    DEVICE_CODE Vector1() noexcept
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal
    /// </summary>
    DEVICE_CODE Vector1(T aVal) noexcept : x(aVal)
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0]]
    /// </summary>
    DEVICE_CODE Vector1(T aVal[1]) noexcept : x(aVal[0])
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal with cast to T
    /// Note: this constructor avoids confusion for integral types when initialising to 0 as it could also be a nullptr
    /// </summary>
    DEVICE_CODE Vector1(int aVal) noexcept
        requires NativeIntegral<T> && (!IsInt<T>)
        : x(T(aVal))
    {
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <Number T2> DEVICE_CODE Vector1(const Vector1<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            Vector1<T2> temp(aVec);
            temp.template ClampToTargetType<T>();
            x = static_cast<T>(temp.x);
        }
        else
        {
            x = static_cast<T>(aVec.x);
        }
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <Number T2> DEVICE_CODE Vector1(Vector1<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            aVec.template ClampToTargetType<T>();
        }
        x = static_cast<T>(aVec.x);
    }

    ~Vector1() = default;

    Vector1(const Vector1 &) noexcept            = default;
    Vector1(Vector1 &&) noexcept                 = default;
    Vector1 &operator=(const Vector1 &) noexcept = default;
    Vector1 &operator=(Vector1 &&) noexcept      = default;
#pragma endregion

#pragma region Operators
    // don't use space-ship operator as it returns true if any comparison returns true.
    // But NPP only returns true if all channels fulfill the comparison.
    // auto operator<=>(const Vector1 &) const = default;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(const Vector1 &aOther) const
        requires RealNumber<T>
    {
        return x < aOther.x;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(const Vector1 &aOther) const
        requires RealNumber<T>
    {
        return x <= aOther.x;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(const Vector1 &aOther) const
        requires RealNumber<T>
    {
        return x > aOther.x;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(const Vector1 &aOther) const
        requires RealNumber<T>
    {
        return x >= aOther.x;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(const Vector1 &aOther) const
    {
        return x == aOther.x;
    }

    /// <summary>
    /// Returns true if any element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(const Vector1 &aOther) const
    {
        return x != aOther.x;
    }

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector1 operator-() const
        requires RealSignedNumber<T> || ComplexNumber<T>
    {
        return Vector1<T>(-x);
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector1 &operator+=(T aOther)
    {
        x += aOther;
        return *this;
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector1 &operator+=(const Vector1 &aOther)
    {
        x += aOther.x;
        return *this;
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector1 operator+(const Vector1 &aOther) const
    {
        return Vector1<T>{T(x + aOther.x)};
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector1 &operator-=(T aOther)
    {
        x -= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector1 &operator-=(const Vector1 &aOther)
    {
        x -= aOther.x;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector1 &SubInv(const Vector1 &aOther)
    {
        x = aOther.x - x;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector1 operator-(const Vector1 &aOther) const
    {
        return Vector1<T>{T(x - aOther.x)};
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector1 &operator*=(T aOther)
    {
        x *= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector1 &operator*=(const Vector1 &aOther)
    {
        x *= aOther.x;
        return *this;
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector1 operator*(const Vector1 &aOther) const
    {
        return Vector1<T>{T(x * aOther.x)};
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector1 &operator/=(T aOther)
    {
        x /= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector1 &operator/=(const Vector1 &aOther)
    {
        x /= aOther.x;
        return *this;
    }

    /// <summary>
    /// Component wise division (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector1 &DivInv(const Vector1 &aOther)
    {
        x = aOther.x / x;
        return *this;
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector1 operator/(const Vector1 &aOther) const
    {
        return Vector1<T>{T(x / aOther.x)};
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T &operator[](Axis1D aAxis) const
        requires DeviceCode<T>
    {
        switch (aAxis)
        {
            case Axis1D::X:
                return x;
        }
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] const T &operator[](Axis1D aAxis) const
        requires HostCode<T>
    {
        switch (aAxis)
        {
            case Axis1D::X:
                return x;
        }

        throw INVALIDARGUMENT(aAxis, aAxis);
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] T &operator[](Axis1D aAxis)
        requires DeviceCode<T>
    {
        switch (aAxis)
        {
            case Axis1D::X:
                return x;
        }
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] T &operator[](Axis1D aAxis)
        requires HostCode<T>
    {
        switch (aAxis)
        {
            case Axis1D::X:
                return x;
        }

        throw INVALIDARGUMENT(aAxis, aAxis);
    }
#pragma region Convert Methods
    /// <summary>
    /// Type conversion without saturation, direct type conversion
    /// </summary>
    template <Number T2> [[nodiscard]] static Vector1<T> DEVICE_CODE Convert(const Vector1<T2> &aVec)
    {
        return {static_cast<T>(aVec.x)};
    }
#pragma endregion

#pragma region Integral only Methods
    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector1<T> &LShift(const Vector1<T> &aOther)
        requires RealIntegral<T>
    {
        x = x << aOther.x;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> LShift(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealIntegral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x << aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector1<T> &RShift(const Vector1<T> &aOther)
        requires RealIntegral<T>
    {
        x = x >> aOther.x;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> RShift(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealIntegral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x >> aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector1<T> &LShift(const T &aOther)
        requires RealIntegral<T>
    {
        x = x << aOther;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> LShift(const Vector1<T> &aLeft, const T &aRight)
        requires RealIntegral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x << aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector1<T> &RShift(const T &aOther)
        requires RealIntegral<T>
    {
        x = x >> aOther;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> RShift(const Vector1<T> &aLeft, const T &aRight)
        requires RealIntegral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x >> aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE Vector1<T> &And(const Vector1<T> &aOther)
        requires RealIntegral<T>
    {
        x = x & aOther.x;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> And(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealIntegral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x & aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE Vector1<T> &Or(const Vector1<T> &aOther)
        requires RealIntegral<T>
    {
        x = x | aOther.x;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Or(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealIntegral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x | aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE Vector1<T> &Xor(const Vector1<T> &aOther)
        requires RealIntegral<T>
    {
        x = x ^ aOther.x;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Xor(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealIntegral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x ^ aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE Vector1<T> &Not()
        requires RealIntegral<T>
    {
        x = ~x;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Not(const Vector1<T> &aVec)
        requires RealIntegral<T>
    {
        Vector1<T> ret;
        ret.x = ~aVec.x;
        return ret;
    }
#pragma endregion

#pragma region Methods
#pragma region Exp
    /// <summary>
    /// Element wise exponential
    /// </summary>
    Vector1<T> &Exp()
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::exp(x);
        return *this;
    }
    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector1<T> &Exp()
        requires NonNativeNumber<T>
    {
        x = T::Exp(x);
        return *this;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector1<T> &Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = exp(x);
        return *this;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Exp(const Vector1<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector1<T> ret;
        ret.x = exp(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Exp(const Vector1<T> &aVec)
        requires HostCode<T> && NativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = std::exp(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Exp(const Vector1<T> &aVec)
        requires NonNativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = T::Exp(aVec.x);
        return ret;
    }
#pragma endregion

#pragma region Log
    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    Vector1<T> &Ln()
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::log(x);
        return *this;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector1<T> &Ln()
        requires NonNativeNumber<T>
    {
        x = T::Ln(x);
        return *this;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector1<T> &Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = log(x);
        return *this;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Ln(const Vector1<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector1<T> ret;
        ret.x = log(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Ln(const Vector1<T> &aVec)
        requires HostCode<T> && NativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = std::log(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Ln(const Vector1<T> &aVec)
        requires NonNativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = T::Ln(aVec.x);
        return ret;
    }
#pragma endregion

#pragma region Sqr
    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE Vector1<T> &Sqr()
    {
        x = x * x;
        return *this;
    }

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Sqr(const Vector1<T> &aVec)
    {
        Vector1<T> ret;
        ret.x = aVec.x * aVec.x;
        return ret;
    }
#pragma endregion

#pragma region Sqrt
    /// <summary>
    /// Element wise square root
    /// </summary>
    Vector1<T> &Sqrt()
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::sqrt(x);
        return *this;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector1<T> &Sqrt()
        requires NonNativeNumber<T>
    {
        x = T::Sqrt(x);
        return *this;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector1<T> &Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = sqrt(x);
        return *this;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Sqrt(const Vector1<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector1<T> ret;
        ret.x = sqrt(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Sqrt(const Vector1<T> &aVec)
        requires HostCode<T> && NativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = std::sqrt(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Sqrt(const Vector1<T> &aVec)
        requires NonNativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = T::Sqrt(aVec.x);
        return ret;
    }
#pragma endregion

#pragma region Abs
    /// <summary>
    /// Element wise absolute
    /// </summary>
    Vector1<T> &Abs()
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        x = std::abs(x);
        return *this;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector1<T> &Abs()
        requires RealSignedNumber<T> && NonNativeNumber<T>
    {
        x = T::Abs(x);
        return *this;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector1<T> &Abs()
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        x = abs(x);
        return *this;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Abs(const Vector1<T> &aVec)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = abs(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Abs(const Vector1<T> &aVec)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = std::abs(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Abs(const Vector1<T> &aVec)
        requires RealSignedNumber<T> && NonNativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = T::Abs(aVec.x);
        return ret;
    }
#pragma endregion

#pragma region AbsDiff
    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector1<T> &AbsDiff(const Vector1<T> &aOther)
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::abs(x - aOther.x);
        return *this;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> AbsDiff(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        Vector1<T> ret;
        ret.x = std::abs(aLeft.x - aRight.x);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector1<T> &AbsDiff(const Vector1<T> &aOther)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = abs(x - aOther.x);
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> AbsDiff(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector1<T> ret;
        ret.x = abs(aLeft.x - aRight.x);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector1<T> &AbsDiff(const Vector1<T> &aOther)
        requires NonNativeNumber<T>
    {
        x = T::Abs(x - aOther.x);
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> AbsDiff(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires NonNativeNumber<T>
    {
        Vector1<T> ret;
        ret.x = T::Abs(aLeft.x - aRight.x);
        return ret;
    }
#pragma endregion

#pragma region Methods for Complex types
    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE Vector1<T> &Conj()
        requires ComplexNumber<T>
    {
        x.Conj();
        return *this;
    }

    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Conj(const Vector1<T> &aValue)
        requires ComplexNumber<T>
    {
        return {T::Conj(aValue.x)};
    }

    /// <summary>
    /// Conjugate complex multiplication: this * conj(aOther)  per element
    /// </summary>
    DEVICE_CODE Vector1<T> &ConjMul(const Vector1<T> &aOther)
        requires ComplexNumber<T>
    {
        x.ConjMul(aOther.x);
        return *this;
    }

    /// <summary>
    /// Conjugate complex multiplication: aLeft * conj(aRight) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> ConjMul(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    {
        return {T::ConjMul(aLeft.x, aRight.x)};
    }

    /// <summary>
    /// Complex magnitude per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector1<complex_basetype_t<T>> Magnitude() const
        requires ComplexFloatingPoint<T>
    {
        Vector1<complex_basetype_t<T>> ret;
        ret.x = x.Magnitude();
        return ret;
    }

    /// <summary>
    /// Complex magnitude squared per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector1<complex_basetype_t<T>> MagnitudeSqr() const
        requires ComplexFloatingPoint<T>
    {
        Vector1<complex_basetype_t<T>> ret;
        ret.x = x.MagnitudeSqr();
        return ret;
    }
#pragma endregion

#pragma region Clamp
    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector1<T> &Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeNumber<T>
    {
        x = max(aMinVal, min(x, aMaxVal));
        return *this;
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    Vector1<T> &Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::max(aMinVal, std::min(x, aMaxVal));
        return *this;
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector1<T> &Clamp(T aMinVal, T aMaxVal)
        requires NonNativeNumber<T>
    {
        x = T::Max(aMinVal, T::Min(x, aMaxVal));
        return *this;
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector1<T> &ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>)
    {
        return Clamp(numeric_limits_conversion<T, TTarget>::lowest(), numeric_limits_conversion<T, TTarget>::max());
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector1<T> &ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>)
    {
        return *this;
    }
#pragma endregion

#pragma region Min
    /// <summary>
    /// Component wise minimum
    /// </summary>
    Vector1<T> &Min(const Vector1<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::min(x, aRight.x);
        return *this;
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector1<T> &Min(const Vector1<T> &aRight)
        requires NonNativeNumber<T>
    {
        x.Min(aRight.x);
        return *this;
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector1<T> &Min(const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        x = min(x, aRight.x);
        return *this;
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Min(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Vector1<T>{T(min(aLeft.x, aRight.x))};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector1<T> Min(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        return Vector1<T>{std::min(aLeft.x, aRight.x)};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Min(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires NonNativeNumber<T>
    {
        return Vector1<T>{T::Min(aLeft.x, aRight.x)};
    }
#pragma endregion

#pragma region Max
    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector1<T> &Max(const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        x = max(x, aRight.x);
        return *this;
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    Vector1<T> &Max(const Vector1<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::max(x, aRight.x);
        return *this;
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector1<T> &Max(const Vector1<T> &aRight)
        requires NonNativeNumber<T>
    {
        x.Max(aRight.x);
        return *this;
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Max(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Vector1<T>{T(max(aLeft.x, aRight.x))};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector1<T> Max(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        return Vector1<T>{std::max(aLeft.x, aRight.x)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Max(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires NonNativeNumber<T>
    {
        return Vector1<T>{T::Max(aLeft.x, aRight.x)};
    }
#pragma endregion

#pragma region Round
    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Round(const Vector1<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Round();
        return ret;
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_ONLY_CODE Vector1<T> &Round()
        requires NonNativeFloatingPoint<T>
    {
        x.Round();
        return *this;
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Vector1<T> &Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = round(x);
        return *this;
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    Vector1<T> &Round()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::round(x);
        return *this;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Floor(const Vector1<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Floor();
        return ret;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE Vector1<T> &Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = floor(x);
        return *this;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    Vector1<T> &Floor()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::floor(x);
        return *this;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE Vector1<T> &Floor()
        requires NonNativeFloatingPoint<T>
    {
        x.Floor();
        return *this;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Ceil(const Vector1<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Ceil();
        return ret;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE Vector1<T> &Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = ceil(x);
        return *this;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    Vector1<T> &Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::ceil(x);
        return *this;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE Vector1<T> &Ceil()
        requires NonNativeFloatingPoint<T>
    {
        x.Ceil();
        return *this;
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> RoundNearest(const Vector1<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.RoundNearest();
        return ret;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector1<T> &RoundNearest()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = __float2int_rn(x);
        return *this;
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    Vector1<T> &RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::nearbyint(x);
        return *this;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE Vector1<T> &RoundNearest()
        requires NonNativeFloatingPoint<T>
    {
        x.RoundNearest();
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> RoundZero(const Vector1<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.RoundZero();
        return ret;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector1<T> &RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = __float2int_rz(x);
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    Vector1<T> &RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::trunc(x);
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE Vector1<T> &RoundZero()
        requires NonNativeFloatingPoint<T>
    {
        x.RoundZero();
        return *this;
    }
#pragma endregion

#pragma region Data accessors
    /// <summary>
    /// Provide a smiliar accessor to inner data as for std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] T *data()
    {
        return &x;
    }

    /// <summary>
    /// Provide a smiliar accessor to inner data as for std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T *data() const
    {
        return &x;
    }
#pragma endregion

#pragma region Compare
    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareEQ(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x == aRight.x) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareGE(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealNumber<T>
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x >= aRight.x) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareGT(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealNumber<T>
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x > aRight.x) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareLE(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealNumber<T>
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x <= aRight.x) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareLT(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealNumber<T>
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x < aRight.x) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareNEQ(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires RealNumber<T>
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x != aRight.x) * TRUE_VALUE;
        return ret;
    }
#pragma endregion
#pragma endregion
};

template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator+(const Vector1<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector1<T>{static_cast<T>(aLeft.x + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator+(T2 aLeft, const Vector1<T> &aRight)
    requires Number<T2>
{
    return Vector1<T>{static_cast<T>(aLeft + aRight.x)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator-(const Vector1<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector1<T>{static_cast<T>(aLeft.x - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator-(T2 aLeft, const Vector1<T> &aRight)
    requires Number<T2>
{
    return Vector1<T>{static_cast<T>(aLeft - aRight.x)};
}

template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator*(const Vector1<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector1<T>{static_cast<T>(aLeft.x * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator*(T2 aLeft, const Vector1<T> &aRight)
    requires Number<T2>
{
    return Vector1<T>{static_cast<T>(aLeft * aRight.x)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator/(const Vector1<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector1<T>{static_cast<T>(aLeft.x / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator/(T2 aLeft, const Vector1<T> &aRight)
    requires Number<T2>
{
    return Vector1<T>{static_cast<T>(aLeft / aRight.x)};
}

template <HostCode T2> std::ostream &operator<<(std::ostream &aOs, const Vector1<T2> &aVec)
{
    aOs << '(' << aVec.x << ')';
    return aOs;
}

template <HostCode T2> std::wostream &operator<<(std::wostream &aOs, const Vector1<T2> &aVec)
{
    aOs << '(' << aVec.x << ')';
    return aOs;
}

template <HostCode T2> std::istream &operator>>(std::istream &aIs, Vector1<T2> &aVec)
{
    aIs >> aVec.x;
    return aIs;
}

template <HostCode T2> std::wistream &operator>>(std::wistream &aIs, Vector1<T2> &aVec)
{
    aIs >> aVec.x;
    return aIs;
}

} // namespace opp
