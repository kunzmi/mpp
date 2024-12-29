#pragma once
#include "complex_typetraits.h"
#include "defines.h"
#include "exception.h"
#include "limits.h"
#include "needSaturationClamp.h"
#include "safeCast.h"
#include "vector_typetraits.h"
#include <cmath>
#include <concepts>
#include <iostream>
#include <type_traits>

namespace opp
{

// forward declaration:
template <ComplexOrNumber T> struct Vector1;
template <ComplexOrNumber T> struct Vector2;
template <ComplexOrNumber T> struct Vector3;
template <ComplexOrNumber T> struct Vector4;

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
template <ComplexOrNumber T> struct alignas(sizeof(T)) Vector1
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
        requires Integral<T> && (!std::same_as<T, int>)
        : x(T(aVal))
    {
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <ComplexOrNumber T2> DEVICE_CODE Vector1(const Vector1<T2> &aVec) noexcept
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
    template <ComplexOrNumber T2> DEVICE_CODE Vector1(Vector1<T2> &aVec) noexcept
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
    {
        return x < aOther.x;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(const Vector1 &aOther) const
    {
        return x <= aOther.x;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(const Vector1 &aOther) const
    {
        return x > aOther.x;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(const Vector1 &aOther) const
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
        requires SignedNumber<T> || ComplexType<T>
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
    template <ComplexOrNumber T2> [[nodiscard]] static Vector1<T> DEVICE_CODE Convert(const Vector1<T2> &aVec)
    {
        return {static_cast<T>(aVec.x)};
    }
#pragma endregion

#pragma region Integral only Methods
    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE void LShift(const Vector1<T> &aOther)
        requires Integral<T>
    {
        x = x << aOther.x;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> LShift(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires Integral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x << aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE void RShift(const Vector1<T> &aOther)
        requires Integral<T>
    {
        x = x >> aOther.x;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> RShift(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires Integral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x >> aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE void LShift(const T &aOther)
        requires Integral<T>
    {
        x = x << aOther;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> LShift(const Vector1<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x << aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE void RShift(const T &aOther)
        requires Integral<T>
    {
        x = x >> aOther;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> RShift(const Vector1<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x >> aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE void And(const Vector1<T> &aOther)
        requires Integral<T>
    {
        x = x & aOther.x;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> And(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires Integral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x & aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE void Or(const Vector1<T> &aOther)
        requires Integral<T>
    {
        x = x | aOther.x;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Or(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires Integral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x | aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE void Xor(const Vector1<T> &aOther)
        requires Integral<T>
    {
        x = x ^ aOther.x;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Xor(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires Integral<T>
    {
        Vector1<T> ret;
        ret.x = aLeft.x ^ aRight.x;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE void Not()
        requires Integral<T>
    {
        x = ~x;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Not(const Vector1<T> &aVec)
        requires Integral<T>
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
    void Exp()
        requires HostCode<T> && NativeType<T>
    {
        x = std::exp(x);
    }
    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE void Exp()
        requires NonNativeType<T>
    {
        x = T::Exp(x);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE void Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = exp(x);
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
        requires HostCode<T> && NativeType<T>
    {
        Vector1<T> ret;
        ret.x = std::exp(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Exp(const Vector1<T> &aVec)
        requires NonNativeType<T>
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
    void Ln()
        requires HostCode<T> && NativeType<T>
    {
        x = std::log(x);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE void Ln()
        requires NonNativeType<T>
    {
        x = T::Ln(x);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE void Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = log(x);
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
        requires HostCode<T> && NativeType<T>
    {
        Vector1<T> ret;
        ret.x = std::log(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Ln(const Vector1<T> &aVec)
        requires NonNativeType<T>
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
    DEVICE_CODE void Sqr()
    {
        x = x * x;
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
    void Sqrt()
        requires HostCode<T> && NativeType<T>
    {
        x = std::sqrt(x);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE void Sqrt()
        requires NonNativeType<T>
    {
        x = T::Sqrt(x);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE void Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = sqrt(x);
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
        requires HostCode<T> && NativeType<T>
    {
        Vector1<T> ret;
        ret.x = std::sqrt(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Sqrt(const Vector1<T> &aVec)
        requires NonNativeType<T>
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
    void Abs()
        requires HostCode<T> && SignedNumber<T> && NativeType<T>
    {
        x = std::abs(x);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE void Abs()
        requires SignedNumber<T> && NonNativeType<T>
    {
        x = T::Abs(x);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE void Abs()
        requires DeviceCode<T> && SignedNumber<T> && NativeType<T>
    {
        x = abs(x);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Abs(const Vector1<T> &aVec)
        requires DeviceCode<T> && SignedNumber<T> && NativeType<T>
    {
        Vector1<T> ret;
        ret.x = abs(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Abs(const Vector1<T> &aVec)
        requires HostCode<T> && SignedNumber<T> && NativeType<T>
    {
        Vector1<T> ret;
        ret.x = std::abs(aVec.x);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Abs(const Vector1<T> &aVec)
        requires NonNativeType<T>
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
    DEVICE_CODE void AbsDiff(const Vector1<T> &aOther)
        requires HostCode<T> && NativeType<T>
    {
        x = std::abs(x - aOther.x);
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> AbsDiff(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        Vector1<T> ret;
        ret.x = std::abs(aLeft.x - aRight.x);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector1<T> &aOther)
        requires DeviceCode<T> && NativeType<T>
    {
        x = abs(x - aOther.x);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> AbsDiff(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        Vector1<T> ret;
        ret.x = abs(aLeft.x - aRight.x);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector1<T> &aOther)
        requires NonNativeType<T>
    {
        x = T::Abs(x - aOther.x);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> AbsDiff(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires NonNativeType<T>
    {
        Vector1<T> ret;
        ret.x = T::Abs(aLeft.x - aRight.x);
        return ret;
    }
#pragma endregion

#pragma region Magnitude(Sqr)
    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        return abs(x);
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] T Magnitude() const
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        return std::abs(x);
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires NonNativeType<T>
    {
        return T::Abs(x);
    }

    /// <summary>
    /// Squared vector length
    /// </summary>
    DEVICE_CODE [[nodiscard]] T MagnitudeSqr() const
        requires FloatingPoint<T>
    {
        return x * x;
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] double Magnitude() const
        requires Integral<T> && DeviceCode<T>
    {
        return abs(x);
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] double Magnitude() const
        requires Integral<T> && HostCode<T>
    {
        return std::abs(x);
    }

    /// <summary>
    /// Squared vector length
    /// </summary>
    [[nodiscard]] double MagnitudeSqr() const
        requires Integral<T> && HostCode<T>
    {
        double dx = to_double(x);

        return dx * dx;
    }
#pragma endregion

#pragma region Normalize
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
    DEVICE_CODE [[nodiscard]] static Vector1<T> Normalize(const Vector1<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Normalize();
        return ret;
    }
#pragma endregion

#pragma region Clamp
    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeType<T>
    {
        x = max(aMinVal, min(x, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    void Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeType<T>
    {
        x = std::max(aMinVal, std::min(x, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires NonNativeType<T>
    {
        x = T::Max(aMinVal, T::Min(x, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>) && (!isSameType<T, HalfFp16> || !isSameType<TTarget, short>)
    {
        Clamp(T(numeric_limits<TTarget>::lowest()), T(numeric_limits<TTarget>::max()));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>) && isSameType<T, HalfFp16> && isSameType<TTarget, short>
    {
        // special case for half floats: the maximum value of short is slightly larger than the closest exact
        // integer in HalfFp16, and as we use round to nearest, the clamping would result in a too large number.
        // Thus for HalfFp16 and short, we clamp to the exact integer smaller than short::max(), i.e. 32752
        constexpr HalfFp16 maxExactShort = HalfFp16::FromUShort(0x77FF); // = 32752
        Clamp(T(numeric_limits<TTarget>::lowest()), maxExactShort);
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>)
    {
    }
#pragma endregion

#pragma region Min
    /// <summary>
    /// Component wise minimum
    /// </summary>
    void Min(const Vector1<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        x = std::min(x, aRight.x);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE void Min(const Vector1<T> &aRight)
        requires NonNativeType<T>
    {
        x.Min(aRight.x);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE void Min(const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        x = min(x, aRight.x);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Min(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        return Vector1<T>{T(min(aLeft.x, aRight.x))};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector1<T> Min(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        return Vector1<T>{std::min(aLeft.x, aRight.x)};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Min(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires NonNativeType<T>
    {
        return Vector1<T>{T::Min(aLeft.x, aRight.x)};
    }
#pragma endregion

#pragma region Max
    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE void Max(const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        x = max(x, aRight.x);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    void Max(const Vector1<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        x = std::max(x, aRight.x);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE void Max(const Vector1<T> &aRight)
        requires NonNativeType<T>
    {
        x.Max(aRight.x);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Max(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        return Vector1<T>{T(max(aLeft.x, aRight.x))};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector1<T> Max(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        return Vector1<T>{std::max(aLeft.x, aRight.x)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Max(const Vector1<T> &aLeft, const Vector1<T> &aRight)
        requires NonNativeType<T>
    {
        return Vector1<T>{T::Max(aLeft.x, aRight.x)};
    }
#pragma endregion

#pragma region Round
    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Round(const Vector1<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Round();
        return ret;
    }

    /// <summary>
    /// Element wise round() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<int> RoundI(const Vector1<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Round();
        return Vector1<int>(ret);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE void Round()
        requires FloatingComplexType<T>
    {
        x.Round();
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE void Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = round(x);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    void Round()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::round(x);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_ONLY_CODE void Round()
        requires NonNativeType<T>
    {
        x.Round();
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Floor(const Vector1<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Floor();
        return ret;
    }

    /// <summary>
    /// Element wise floor() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<int> FloorI(const Vector1<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Floor();
        return Vector1<int>(ret);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE void Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = floor(x);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    void Floor()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::floor(x);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE void Floor()
        requires NonNativeType<T>
    {
        x.Floor();
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> Ceil(const Vector1<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Ceil();
        return ret;
    }

    /// <summary>
    /// Element wise ceil() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<int> CeilI(const Vector1<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.Ceil();
        return Vector1<int>(ret);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE void Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = ceil(x);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    void Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::ceil(x);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE void Ceil()
        requires NonNativeType<T>
    {
        x.Ceil();
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> RoundNearest(const Vector1<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.RoundNearest();
        return ret;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE void RoundNearest()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = __float2int_rn(x);
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    void RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::nearbyint(x);
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE void RoundNearest()
        requires NonNativeType<T>
    {
        x.RoundNearest();
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<T> RoundZero(const Vector1<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector1<T> ret = aValue;
        ret.RoundZero();
        return ret;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE void RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = __float2int_rz(x);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    void RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::trunc(x);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE void RoundZero()
        requires NonNativeType<T>
    {
        x.RoundZero();
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
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x >= aRight.x) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareGT(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x > aRight.x) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareLE(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x <= aRight.x) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareLT(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    {
        Vector1<byte> ret;
        ret.x = byte(aLeft.x < aRight.x) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector1<byte> CompareNEQ(const Vector1<T> &aLeft, const Vector1<T> &aRight)
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
    requires ComplexOrNumber<T2>
{
    return Vector1<T>{T(aLeft.x + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator+(T2 aLeft, const Vector1<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector1<T>{T(aLeft + aRight.x)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator-(const Vector1<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector1<T>{T(aLeft.x - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator-(T2 aLeft, const Vector1<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector1<T>{T(aLeft - aRight.x)};
}

template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator*(const Vector1<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector1<T>{T(aLeft.x * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator*(T2 aLeft, const Vector1<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector1<T>{T(aLeft * aRight.x)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator/(const Vector1<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector1<T>{T(aLeft.x / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector1<T> operator/(T2 aLeft, const Vector1<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector1<T>{T(aLeft / aRight.x)};
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
