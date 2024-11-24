#pragma once
#include "complex_typetraits.h"
#include "defines.h"
#include "exception.h"
#include "limits.h"
#include "needSaturationClamp.h"
#include "safeCast.h"
#include "vector_typetraits.h"
#include "vector3.h" //for additional constructor from Vector3<T>
#include "vector4.h"
#include <cmath>
#include <concepts>
#include <iostream>
#include <type_traits>

namespace opp
{

// forward declaration:
template <ComplexOrNumber T> struct Vector1;
template <ComplexOrNumber T> struct Vector2;

/// <summary>
/// A four T component vector. Operations are performed on the first three channels, W is treated as additional Alpha
/// channel and remains unused. Can replace CUDA's vector4 types
/// </summary>
template <ComplexOrNumber T> struct alignas(4 * sizeof(T)) Vector4A
{
    T x;
    T y;
    T z;
    T w;

    /// <summary>
    /// Default constructor does not initializes the members
    /// </summary>
    DEVICE_CODE Vector4A() noexcept
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal, except w
    /// </summary>
    DEVICE_CODE explicit Vector4A(T aVal) noexcept : x(aVal), y(aVal), z(aVal)
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0], aVal[1], aVal[2]], w remains unitialized
    /// </summary>
    DEVICE_CODE explicit Vector4A(T aVal[3]) noexcept : x(aVal[0]), y(aVal[1]), z(aVal[2])
    {
    }

    /// <summary>
    /// Initializes vector to [aX, aY, aZ], w remains unitialized
    /// </summary>
    DEVICE_CODE Vector4A(T aX, T aY, T aZ) noexcept : x(aX), y(aY), z(aZ)
    {
    }

    /// <summary>
    /// Usefull constructor if we want a Vector4A from 3 channel pixel Vector3, w remains unitialized
    /// </summary>
    DEVICE_CODE explicit Vector4A(const Vector3<T> &aVec3) noexcept : x(aVec3.x), y(aVec3.y), z(aVec3.z)
    {
    }

    /// <summary>
    /// Usefull constructor if we want a Vector4A from 4 channel pixel Vector4, w remains unitialized
    /// </summary>
    DEVICE_CODE explicit Vector4A(const Vector4<T> &aVec4) noexcept : x(aVec4.x), y(aVec4.y), z(aVec4.z)
    {
    }

    /// <summary>
    /// Type conversion with saturation if needed, w remains unitialized<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <ComplexOrNumber T2> DEVICE_CODE explicit Vector4A(const Vector4A<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            Vector4A<T2> temp(aVec);
            temp.template ClampToTargetType<T>();
            x = static_cast<T>(temp.x);
            y = static_cast<T>(temp.y);
            z = static_cast<T>(temp.z);
        }
        else
        {
            x = static_cast<T>(aVec.x);
            y = static_cast<T>(aVec.y);
            z = static_cast<T>(aVec.z);
        }
    }

    /// <summary>
    /// Type conversion with saturation if needed, w remains unitialized<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <ComplexOrNumber T2> DEVICE_CODE explicit Vector4A(Vector4A<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            aVec.template ClampToTargetType<T>();
        }
        x = static_cast<T>(aVec.x);
        y = static_cast<T>(aVec.y);
        z = static_cast<T>(aVec.z);
    }

    ~Vector4A() = default;

    Vector4A(const Vector4A &) noexcept            = default;
    Vector4A(Vector4A &&) noexcept                 = default;
    Vector4A &operator=(const Vector4A &) noexcept = default;
    Vector4A &operator=(Vector4A &&) noexcept      = default;

    // don't use space-ship operator as it returns true if any comparison returns true.
    // But NPP only returns true if all channels fulfill the comparison.
    // Also return TRUE_VALUE as byte instead of bool
    // auto operator<=>(const Vector4A &) const = default;

    /// <summary>
    /// Returns true-value if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    byte operator<(const Vector4A &aOther) const
    {
        bool res = x < aOther.x;
        res &= y < aOther.y;
        res &= z < aOther.z;
        return res * TRUE_VALUE;
    }

    /// <summary>
    /// Returns true-value if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    byte operator<=(const Vector4A &aOther) const
    {
        bool res = x <= aOther.x;
        res &= y <= aOther.y;
        res &= z <= aOther.z;
        return res * TRUE_VALUE;
    }

    /// <summary>
    /// Returns true-value if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    byte operator>(const Vector4A &aOther) const
    {
        bool res = x > aOther.x;
        res &= y > aOther.y;
        res &= z > aOther.z;
        return res * TRUE_VALUE;
    }

    /// <summary>
    /// Returns true-value if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    byte operator>=(const Vector4A &aOther) const
    {
        bool res = x >= aOther.x;
        res &= y >= aOther.y;
        res &= z >= aOther.z;
        return res * TRUE_VALUE;
    }

    /// <summary>
    /// Returns true-value if each element comparison is true, ignoring alpha / w-value
    /// </summary>
    byte operator==(const Vector4A &aOther) const
    {
        bool res = x == aOther.x;
        res &= y == aOther.y;
        res &= z == aOther.z;
        return res * TRUE_VALUE;
    }

    /// <summary>
    /// Returns true-value if any element comparison is true, ignoring alpha / w-value
    /// </summary>
    byte operator!=(const Vector4A &aOther) const
    {
        bool res = x != aOther.x;
        res |= y != aOther.y;
        res |= z != aOther.z;
        return res * TRUE_VALUE;
    }

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator-() const
        requires SignedNumber<T>
    {
        return Vector4A<T>(T(-x), T(-y), T(-z));
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector4A &operator+=(T aOther)
    {
        x += aOther;
        y += aOther;
        z += aOther;
        return *this;
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector4A &operator+=(const Vector4A &aOther)
    {
        x += aOther.x;
        y += aOther.y;
        z += aOther.z;
        return *this;
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator+(const Vector4A &aOther) const
    {
        return Vector4A<T>{T(x + aOther.x), T(y + aOther.y), T(z + aOther.z)};
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector4A &operator-=(T aOther)
    {
        x -= aOther;
        y -= aOther;
        z -= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector4A &operator-=(const Vector4A &aOther)
    {
        x -= aOther.x;
        y -= aOther.y;
        z -= aOther.z;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator-(const Vector4A &aOther) const
    {
        return Vector4A<T>{T(x - aOther.x), T(y - aOther.y), T(z - aOther.z)};
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector4A &operator*=(T aOther)
    {
        x *= aOther;
        y *= aOther;
        z *= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector4A &operator*=(const Vector4A &aOther)
    {
        x *= aOther.x;
        y *= aOther.y;
        z *= aOther.z;
        return *this;
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator*(const Vector4A &aOther) const
    {
        return Vector4A<T>{T(x * aOther.x), T(y * aOther.y), T(z * aOther.z)};
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector4A &operator/=(T aOther)
    {
        x /= aOther;
        y /= aOther;
        z /= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector4A &operator/=(const Vector4A &aOther)
    {
        x /= aOther.x;
        y /= aOther.y;
        z /= aOther.z;
        return *this;
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A operator/(const Vector4A &aOther) const
    {
        return Vector4A<T>{T(x / aOther.x), T(y / aOther.y), T(z / aOther.z)};
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T &operator[](Axis4D aAxis) const
        requires DeviceCode<T>
    {
        switch (aAxis)
        {
            case Axis4D::X:
                return x;
            case Axis4D::Y:
                return y;
            case Axis4D::Z:
                return z;
            case Axis4D::W:
                return w;
        }
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] const T &operator[](Axis4D aAxis) const
        requires HostCode<T>
    {
        switch (aAxis)
        {
            case Axis4D::X:
                return x;
            case Axis4D::Y:
                return y;
            case Axis4D::Z:
                return z;
            case Axis4D::W:
                return w;
        }

        throw INVALIDARGUMENT(aAxis, aAxis);
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] T &operator[](Axis4D aAxis)
        requires DeviceCode<T>
    {
        switch (aAxis)
        {
            case Axis4D::X:
                return x;
            case Axis4D::Y:
                return y;
            case Axis4D::Z:
                return z;
            case Axis4D::W:
                return w;
        }
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] T &operator[](Axis4D aAxis)
        requires HostCode<T>
    {
        switch (aAxis)
        {
            case Axis4D::X:
                return x;
            case Axis4D::Y:
                return y;
            case Axis4D::Z:
                return z;
            case Axis4D::W:
                return w;
        }

        throw INVALIDARGUMENT(aAxis, aAxis);
    }

    /// <summary>
    /// Type conversion without saturation, direct type conversion
    /// </summary>
    template <ComplexOrNumber T2> [[nodiscard]] static Vector4A<T> DEVICE_CODE Convert(const Vector4A<T2> &aVec)
    {
        return {static_cast<T>(aVec.x), static_cast<T>(aVec.y), static_cast<T>(aVec.z)};
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE void LShift(const Vector4A<T> &aOther)
        requires Integral<T>
    {
        x = x << aOther.x;
        y = y << aOther.y;
        z = z << aOther.z;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> LShift(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires Integral<T>
    {
        Vector4A<T> ret;
        ret.x = aLeft.x << aRight.x;
        ret.y = aLeft.y << aRight.y;
        ret.z = aLeft.z << aRight.z;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE void RShift(const Vector4A<T> &aOther)
        requires Integral<T>
    {
        x = x >> aOther.x;
        y = y >> aOther.y;
        z = z >> aOther.z;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> RShift(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires Integral<T>
    {
        Vector4A<T> ret;
        ret.x = aLeft.x >> aRight.x;
        ret.y = aLeft.y >> aRight.y;
        ret.z = aLeft.z >> aRight.z;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE void LShift(const T &aOther)
        requires Integral<T>
    {
        x = x << aOther;
        y = y << aOther;
        z = z << aOther;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> LShift(const Vector4A<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector4A<T> ret;
        ret.x = aLeft.x << aRight;
        ret.y = aLeft.y << aRight;
        ret.z = aLeft.z << aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE void RShift(const T &aOther)
        requires Integral<T>
    {
        x = x >> aOther;
        y = y >> aOther;
        z = z >> aOther;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> RShift(const Vector4A<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector4A<T> ret;
        ret.x = aLeft.x >> aRight;
        ret.y = aLeft.y >> aRight;
        ret.z = aLeft.z >> aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE void And(const Vector4A<T> &aOther)
        requires Integral<T>
    {
        x = x & aOther.x;
        y = y & aOther.y;
        z = z & aOther.z;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> And(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires Integral<T>
    {
        Vector4A<T> ret;
        ret.x = aLeft.x & aRight.x;
        ret.y = aLeft.y & aRight.y;
        ret.z = aLeft.z & aRight.z;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE void Or(const Vector4A<T> &aOther)
        requires Integral<T>
    {
        x = x | aOther.x;
        y = y | aOther.y;
        z = z | aOther.z;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Or(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires Integral<T>
    {
        Vector4A<T> ret;
        ret.x = aLeft.x | aRight.x;
        ret.y = aLeft.y | aRight.y;
        ret.z = aLeft.z | aRight.z;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE void Xor(const Vector4A<T> &aOther)
        requires Integral<T>
    {
        x = x ^ aOther.x;
        y = y ^ aOther.y;
        z = z ^ aOther.z;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Xor(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires Integral<T>
    {
        Vector4A<T> ret;
        ret.x = aLeft.x ^ aRight.x;
        ret.y = aLeft.y ^ aRight.y;
        ret.z = aLeft.z ^ aRight.z;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE void Not()
        requires Integral<T>
    {
        x = ~x;
        y = ~y;
        z = ~z;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Not(const Vector4A<T> &aVec)
        requires Integral<T>
    {
        Vector4A<T> ret;
        ret.x = ~aVec.x;
        ret.y = ~aVec.y;
        ret.z = ~aVec.z;
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    void Exp()
        requires HostCode<T>
    {
        x = std::exp(x);
        y = std::exp(y);
        z = std::exp(z);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE void Exp()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = exp(x);
        y = exp(y);
        z = exp(z);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Exp(const Vector4A<T> &aVec)
        requires DeviceCode<T> && FloatingPoint<T>
    {
        Vector4A<T> ret;
        ret.x = exp(aVec.x);
        ret.y = exp(aVec.y);
        ret.z = exp(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Exp(const Vector4A<T> &aVec)
        requires HostCode<T>
    {
        Vector4A<T> ret;
        ret.x = std::exp(aVec.x);
        ret.y = std::exp(aVec.y);
        ret.z = std::exp(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    void Ln()
        requires HostCode<T>
    {
        x = std::log(x);
        y = std::log(y);
        z = std::log(z);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE void Ln()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = log(x);
        y = log(y);
        z = log(z);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Ln(const Vector4A<T> &aVec)
        requires DeviceCode<T> && FloatingPoint<T>
    {
        Vector4A<T> ret;
        ret.x = log(aVec.x);
        ret.y = log(aVec.y);
        ret.z = log(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Ln(const Vector4A<T> &aVec)
        requires HostCode<T>
    {
        Vector4A<T> ret;
        ret.x = std::log(aVec.x);
        ret.y = std::log(aVec.y);
        ret.z = std::log(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE void Sqr()
    {
        x = x * x;
        y = y * y;
        z = z * z;
    }

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Sqr(const Vector4A<T> &aVec)
    {
        Vector4A<T> ret;
        ret.x = aVec.x * aVec.x;
        ret.y = aVec.y * aVec.y;
        ret.z = aVec.z * aVec.z;
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    void Sqrt()
        requires HostCode<T>
    {
        x = std::sqrt(x);
        y = std::sqrt(y);
        z = std::sqrt(z);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE void Sqrt()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = sqrt(x);
        y = sqrt(y);
        z = sqrt(z);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Sqrt(const Vector4A<T> &aVec)
        requires DeviceCode<T> && FloatingPoint<T>
    {
        Vector4A<T> ret;
        ret.x = sqrt(aVec.x);
        ret.y = sqrt(aVec.y);
        ret.z = sqrt(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Sqrt(const Vector4A<T> &aVec)
        requires HostCode<T>
    {
        Vector4A<T> ret;
        ret.x = std::sqrt(aVec.x);
        ret.y = std::sqrt(aVec.y);
        ret.z = std::sqrt(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    void Abs()
        requires HostCode<T>
    {
        x = std::abs(x);
        y = std::abs(y);
        z = std::abs(z);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE void Abs()
        requires DeviceCode<T>
    {
        x = abs(x);
        y = abs(y);
        z = abs(z);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Abs(const Vector4A<T> &aVec)
        requires DeviceCode<T>
    {
        Vector4A<T> ret;
        ret.x = abs(aVec.x);
        ret.y = abs(aVec.y);
        ret.z = abs(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Abs(const Vector4A<T> &aVec)
        requires HostCode<T>
    {
        Vector4A<T> ret;
        ret.x = std::abs(aVec.x);
        ret.y = std::abs(aVec.y);
        ret.z = std::abs(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector4A<T> &aOther)
        requires HostCode<T>
    {
        x = std::abs(x - aOther.x);
        y = std::abs(y - aOther.y);
        z = std::abs(z - aOther.z);
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires HostCode<T>
    {
        Vector4A<T> ret;
        ret.x = std::abs(aLeft.x - aRight.x);
        ret.y = std::abs(aLeft.y - aRight.y);
        ret.z = std::abs(aLeft.z - aRight.z);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector4A<T> &aOther)
        requires DeviceCode<T>
    {
        x = abs(x - aOther.x);
        y = abs(y - aOther.y);
        z = abs(z - aOther.z);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> AbsDiff(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires DeviceCode<T>
    {
        Vector4A<T> ret;
        ret.x = abs(aLeft.x - aRight.x);
        ret.y = abs(aLeft.y - aRight.y);
        ret.z = abs(aLeft.z - aRight.z);
        return ret;
    }

    /// <summary>
    /// Vector dot product
    /// </summary>
    DEVICE_CODE [[nodiscard]] static T Dot(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
        requires FloatingPoint<T>
    {
        return aLeft.x * aRight.x + aLeft.y * aRight.y + aLeft.z * aRight.z;
    }

    /// <summary>
    /// Vector dot product
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Dot(const Vector4A<T> &aRight) const
        requires FloatingPoint<T>
    {
        return x * aRight.x + y * aRight.y + z * aRight.z;
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires DeviceCode<T> && FloatingPoint<T>
    {
        return sqrt(Dot(*this, *this));
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] T Magnitude() const
        requires HostCode<T> && FloatingPoint<T>
    {
        return std::sqrt(Dot(*this, *this));
    }

    /// <summary>
    /// Squared vector length
    /// </summary>
    DEVICE_CODE [[nodiscard]] T MagnitudeSqr() const
        requires FloatingPoint<T>
    {
        return Dot(*this, *this);
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] double Magnitude() const
        requires Integral<T> && DeviceCode<T>
    {
        return sqrt(MagnitudeSqr());
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] double Magnitude() const
        requires Integral<T> && HostCode<T>
    {
        return std::sqrt(MagnitudeSqr());
    }

    /// <summary>
    /// Squared vector length
    /// </summary>
    [[nodiscard]] double MagnitudeSqr() const
        requires Integral<T> && HostCode<T>
    {
        double dx = to_double(x);
        double dy = to_double(y);
        double dz = to_double(z);

        return dx * dx + dy * dy + dz * dz;
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
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Normalize(const Vector4A<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4A<T> ret = aValue;
        ret.Normalize();
        return ret;
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T>
    {
        x = max(aMinVal, min(x, aMaxVal));
        y = max(aMinVal, min(y, aMaxVal));
        z = max(aMinVal, min(z, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    void Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T>
    {
        x = std::max(aMinVal, std::min(x, aMaxVal));
        y = std::max(aMinVal, std::min(y, aMaxVal));
        z = std::max(aMinVal, std::min(z, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>)
    {
        Clamp(T(numeric_limits<TTarget>::lowest()), T(numeric_limits<TTarget>::max()));
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

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] Vector4A<T> Min(const Vector4A<T> &aRight) const
        requires HostCode<T>
    {
        return Vector4A<T>{std::min(x, aRight.x), std::min(y, aRight.y), std::min(z, aRight.z)};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A<T> Min(const Vector4A<T> &aRight) const
        requires DeviceCode<T>
    {
        return Vector4A<T>{min(x, aRight.x), min(y, aRight.y), min(z, aRight.z)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector4A<T> Max(const Vector4A<T> &aRight) const
        requires DeviceCode<T>
    {
        return Vector4A<T>{max(x, aRight.x), max(y, aRight.y), max(z, aRight.z)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] Vector4A<T> Max(const Vector4A<T> &aRight) const
        requires HostCode<T>
    {
        return Vector4A<T>{std::max(x, aRight.x), std::max(y, aRight.y), std::max(z, aRight.z)};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Min(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
    {
        return aLeft.Min(aRight);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Max(const Vector4A<T> &aLeft, const Vector4A<T> &aRight)
    {
        return aLeft.Max(aRight);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T>
    {
        return min(min(x, y), z);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    [[nodiscard]] T Min() const
        requires HostCode<T>
    {
        return std::min({x, y, z});
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T>
    {
        return max(max(x, y), z);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    [[nodiscard]] T Max() const
        requires HostCode<T>
    {
        return std::max({x, y, z});
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Round(const Vector4A<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4A<T> ret = aValue;
        ret.Round();
        return ret;
    }

    /// <summary>
    /// Element wise round() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<int> RoundI(const Vector4A<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4A<T> ret = aValue;
        ret.Round();
        return Vector4A<int>(ret);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE void Round()
        requires FloatingComplexType<T>
    {
        x.Round();
        y.Round();
        z.Round();
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE void Round()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = round(x);
        y = round(y);
        z = round(z);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    void Round()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::round(x);
        y = std::round(y);
        z = std::round(z);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Floor(const Vector4A<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4A<T> ret = aValue;
        ret.Floor();
        return ret;
    }

    /// <summary>
    /// Element wise floor() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<int> FloorI(const Vector4A<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4A<T> ret = aValue;
        ret.Floor();
        return Vector4A<int>(ret);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE void Floor()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = floor(x);
        y = floor(y);
        z = floor(z);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    void Floor()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::floor(x);
        y = std::floor(y);
        z = std::floor(z);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> Ceil(const Vector4A<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4A<T> ret = aValue;
        ret.Ceil();
        return ret;
    }

    /// <summary>
    /// Element wise ceil() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<int> CeilI(const Vector4A<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4A<T> ret = aValue;
        ret.Ceil();
        return Vector4A<int>(ret);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE void Ceil()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = ceil(x);
        y = ceil(y);
        z = ceil(z);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    void Ceil()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::ceil(x);
        y = std::ceil(y);
        z = std::ceil(z);
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> RoundNearest(const Vector4A<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4A<T> ret = aValue;
        ret.RoundNearest();
        return ret;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE void RoundNearest()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = __float2int_rn(x);
        y = __float2int_rn(y);
        z = __float2int_rn(z);
    }

    /// <summary>
    /// Element wise round nearest ties to even <para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    void RoundNearest()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::nearbyint(x);
        y = std::nearbyint(y);
        z = std::nearbyint(z);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector4A<T> RoundZero(const Vector4A<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector4A<T> ret = aValue;
        ret.RoundZero();
        return ret;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE void RoundZero()
        requires DeviceCode<T> && FloatingPoint<T>
    {
        x = __float2int_rz(x);
        y = __float2int_rz(y);
        z = __float2int_rz(z);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    void RoundZero()
        requires HostCode<T> && FloatingPoint<T>
    {
        x = std::trunc(x);
        y = std::trunc(y);
        z = std::trunc(z);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<T> XYZ() const
    {
        return Vector3<T>(x, y, z);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<T> YZW() const
    {
        return Vector3<T>(y, z, w);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> XY() const
    {
        return Vector2<T>(x, y);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> YZ() const
    {
        return Vector2<T>(y, z);
    }

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> ZW() const
    {
        return Vector2<T>(z, w);
    }
};

template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator+(const Vector4A<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4A<T>{T(aLeft.x + aRight), T(aLeft.y + aRight), T(aLeft.z + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator+(T2 aLeft, const Vector4A<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4A<T>{T(aLeft + aRight.x), T(aLeft + aRight.y), T(aLeft + aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator-(const Vector4A<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4A<T>{T(aLeft.x - aRight), T(aLeft.y - aRight), T(aLeft.z - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator-(T2 aLeft, const Vector4A<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4A<T>{T(aLeft - aRight.x), T(aLeft - aRight.y), T(aLeft - aRight.z)};
}

template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator*(const Vector4A<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4A<T>{T(aLeft.x * aRight), T(aLeft.y * aRight), T(aLeft.z * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator*(T2 aLeft, const Vector4A<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4A<T>{T(aLeft * aRight.x), T(aLeft * aRight.y), T(aLeft * aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator/(const Vector4A<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4A<T>{T(aLeft.x / aRight), T(aLeft.y / aRight), T(aLeft.z / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector4A<T> operator/(T2 aLeft, const Vector4A<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector4A<T>{T(aLeft / aRight.x), T(aLeft / aRight.y), T(aLeft / aRight.z)};
}

template <HostCode T2> std::ostream &operator<<(std::ostream &aOs, const Vector4A<T2> &aVec)
{
    aOs << '(' << aVec.x << ", " << aVec.y << ", " << aVec.z << ", " << aVec.w << ')';
    return aOs;
}

template <HostCode T2> std::wostream &operator<<(std::wostream &aOs, const Vector4A<T2> &aVec)
{
    aOs << '(' << aVec.x << ", " << aVec.y << ", " << aVec.z << ", " << aVec.w << ')';
    return aOs;
}

template <HostCode T2> std::istream &operator>>(std::istream &aIs, Vector4A<T2> &aVec)
{
    aIs >> aVec.x >> aVec.y >> aVec.z >> aVec.w;
    return aIs;
}

template <HostCode T2> std::wistream &operator>>(std::wistream &aIs, Vector4A<T2> &aVec)
{
    aIs >> aVec.x >> aVec.y >> aVec.z >> aVec.w;
    return aIs;
}

template <ComplexOrNumber T> Vector4<T> &Vector4<T>::operator=(const Vector4A<T> &aOther) noexcept
{
    x = aOther.x;
    y = aOther.y;
    z = aOther.z;
    return *this;
}
} // namespace opp
