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

enum class Axis3D
{
    X = 0,
    Y = 1,
    Z = 2
};

inline std::ostream &operator<<(std::ostream &aOs, const Axis3D &aAxis)
{
    switch (aAxis)
    {
        case Axis3D::X:
            aOs << 'X';
            return aOs;
        case Axis3D::Y:
            aOs << 'Y';
            return aOs;
        case Axis3D::Z:
            aOs << 'Z';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X, Y or Z (0, 1 or 2).";
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const Axis3D &aAxis)
{
    switch (aAxis)
    {
        case Axis3D::X:
            aOs << 'X';
            return aOs;
        case Axis3D::Y:
            aOs << 'Y';
            return aOs;
        case Axis3D::Z:
            aOs << 'Z';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X, Y or Z (0, 1 or 2).";
    return aOs;
}

/// <summary>
/// A three T component vector. Can replace CUDA's vector3 types
/// </summary>
template <ComplexOrNumber T> struct alignas(sizeof(T)) Vector3
{
    T x;
    T y;
    T z;

    template <typename T2> struct same_vector_size_different_type
    {
        using vector = Vector3<T2>;
    };

    template <typename T2>
    using same_vector_size_different_type_t = typename same_vector_size_different_type<T2>::vector;

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    DEVICE_CODE Vector3() noexcept
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal
    /// </summary>
    DEVICE_CODE Vector3(T aVal) noexcept : x(aVal), y(aVal), z(aVal)
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0], aVal[1], aVal[2]]
    /// </summary>
    DEVICE_CODE Vector3(T aVal[3]) noexcept : x(aVal[0]), y(aVal[1]), z(aVal[2])
    {
    }

    /// <summary>
    /// Initializes vector to [aX, aY, aZ]
    /// </summary>
    DEVICE_CODE Vector3(T aX, T aY, T aZ) noexcept : x(aX), y(aY), z(aZ)
    {
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <ComplexOrNumber T2> DEVICE_CODE Vector3(const Vector3<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            Vector3<T2> temp(aVec);
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
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <ComplexOrNumber T2> DEVICE_CODE Vector3(Vector3<T2> &aVec) noexcept
    {
        if constexpr (need_saturation_clamp_v<T2, T>)
        {
            aVec.template ClampToTargetType<T>();
        }
        x = static_cast<T>(aVec.x);
        y = static_cast<T>(aVec.y);
        z = static_cast<T>(aVec.z);
    }

    ~Vector3() = default;

    Vector3(const Vector3 &) noexcept            = default;
    Vector3(Vector3 &&) noexcept                 = default;
    Vector3 &operator=(const Vector3 &) noexcept = default;
    Vector3 &operator=(Vector3 &&) noexcept      = default;
#pragma endregion

#pragma region Operators
    // don't use space-ship operator as it returns true if any comparison returns true.
    // But NPP only returns true if all channels fulfill the comparison.
    // auto operator<=>(const Vector3 &) const = default;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(const Vector3 &aOther) const
    {
        bool res = x < aOther.x;
        res &= y < aOther.y;
        res &= z < aOther.z;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(const Vector3 &aOther) const
    {
        bool res = x <= aOther.x;
        res &= y <= aOther.y;
        res &= z <= aOther.z;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(const Vector3 &aOther) const
    {
        bool res = x > aOther.x;
        res &= y > aOther.y;
        res &= z > aOther.z;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(const Vector3 &aOther) const
    {
        bool res = x >= aOther.x;
        res &= y >= aOther.y;
        res &= z >= aOther.z;
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(const Vector3 &aOther) const
    {
        bool res = x == aOther.x;
        res &= y == aOther.y;
        res &= z == aOther.z;
        return res;
    }

    /// <summary>
    /// Returns true if any element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(const Vector3 &aOther) const
    {
        bool res = x != aOther.x;
        res |= y != aOther.y;
        res |= z != aOther.z;
        return res;
    }

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator-() const
        requires SignedNumber<T> || ComplexType<T>
    {
        return Vector3<T>(-x, -y, -z);
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector3 &operator+=(T aOther)
    {
        x += aOther;
        y += aOther;
        z += aOther;
        return *this;
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector3 &operator+=(const Vector3 &aOther)
    {
        x += aOther.x;
        y += aOther.y;
        z += aOther.z;
        return *this;
    }

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator+(const Vector3 &aOther) const
    {
        return Vector3<T>{T(x + aOther.x), T(y + aOther.y), T(z + aOther.z)};
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector3 &operator-=(T aOther)
    {
        x -= aOther;
        y -= aOther;
        z -= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector3 &operator-=(const Vector3 &aOther)
    {
        x -= aOther.x;
        y -= aOther.y;
        z -= aOther.z;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector3 &SubInv(const Vector3 &aOther)
    {
        x = aOther.x - x;
        y = aOther.y - y;
        z = aOther.z - z;
        return *this;
    }

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator-(const Vector3 &aOther) const
    {
        return Vector3<T>{T(x - aOther.x), T(y - aOther.y), T(z - aOther.z)};
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector3 &operator*=(T aOther)
    {
        x *= aOther;
        y *= aOther;
        z *= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector3 &operator*=(const Vector3 &aOther)
    {
        x *= aOther.x;
        y *= aOther.y;
        z *= aOther.z;
        return *this;
    }

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator*(const Vector3 &aOther) const
    {
        return Vector3<T>{T(x * aOther.x), T(y * aOther.y), T(z * aOther.z)};
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector3 &operator/=(T aOther)
    {
        x /= aOther;
        y /= aOther;
        z /= aOther;
        return *this;
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector3 &operator/=(const Vector3 &aOther)
    {
        x /= aOther.x;
        y /= aOther.y;
        z /= aOther.z;
        return *this;
    }

    /// <summary>
    /// Component wise division (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInv(const Vector3 &aOther)
    {
        x = aOther.x / x;
        y = aOther.y / y;
        z = aOther.z / z;
        return *this;
    }

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator/(const Vector3 &aOther) const
    {
        return Vector3<T>{T(x / aOther.x), T(y / aOther.y), T(z / aOther.z)};
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T &operator[](Axis3D aAxis) const
        requires DeviceCode<T>
    {
        switch (aAxis)
        {
            case Axis3D::X:
                return x;
            case Axis3D::Y:
                return y;
            case Axis3D::Z:
                return z;
        }
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] const T &operator[](Axis3D aAxis) const
        requires HostCode<T>
    {
        switch (aAxis)
        {
            case Axis3D::X:
                return x;
            case Axis3D::Y:
                return y;
            case Axis3D::Z:
                return z;
        }

        throw INVALIDARGUMENT(aAxis, aAxis);
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] T &operator[](Axis3D aAxis)
        requires DeviceCode<T>
    {
        switch (aAxis)
        {
            case Axis3D::X:
                return x;
            case Axis3D::Y:
                return y;
            case Axis3D::Z:
                return z;
        }
    }

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] T &operator[](Axis3D aAxis)
        requires HostCode<T>
    {
        switch (aAxis)
        {
            case Axis3D::X:
                return x;
            case Axis3D::Y:
                return y;
            case Axis3D::Z:
                return z;
        }

        throw INVALIDARGUMENT(aAxis, aAxis);
    }
#pragma endregion

#pragma region Convert Methods
    /// <summary>
    /// Type conversion without saturation, direct type conversion
    /// </summary>
    template <ComplexOrNumber T2> [[nodiscard]] static Vector3<T> DEVICE_CODE Convert(const Vector3<T2> &aVec)
    {
        return {static_cast<T>(aVec.x), static_cast<T>(aVec.y), static_cast<T>(aVec.z)};
    }
#pragma endregion

#pragma region Integral only Methods
    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE void LShift(const Vector3<T> &aOther)
        requires Integral<T>
    {
        x = x << aOther.x;
        y = y << aOther.y;
        z = z << aOther.z;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> LShift(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires Integral<T>
    {
        Vector3<T> ret;
        ret.x = aLeft.x << aRight.x;
        ret.y = aLeft.y << aRight.y;
        ret.z = aLeft.z << aRight.z;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE void RShift(const Vector3<T> &aOther)
        requires Integral<T>
    {
        x = x >> aOther.x;
        y = y >> aOther.y;
        z = z >> aOther.z;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RShift(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires Integral<T>
    {
        Vector3<T> ret;
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
    DEVICE_CODE [[nodiscard]] static Vector3<T> LShift(const Vector3<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector3<T> ret;
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
    DEVICE_CODE [[nodiscard]] static Vector3<T> RShift(const Vector3<T> &aLeft, const T &aRight)
        requires Integral<T>
    {
        Vector3<T> ret;
        ret.x = aLeft.x >> aRight;
        ret.y = aLeft.y >> aRight;
        ret.z = aLeft.z >> aRight;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE void And(const Vector3<T> &aOther)
        requires Integral<T>
    {
        x = x & aOther.x;
        y = y & aOther.y;
        z = z & aOther.z;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> And(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires Integral<T>
    {
        Vector3<T> ret;
        ret.x = aLeft.x & aRight.x;
        ret.y = aLeft.y & aRight.y;
        ret.z = aLeft.z & aRight.z;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE void Or(const Vector3<T> &aOther)
        requires Integral<T>
    {
        x = x | aOther.x;
        y = y | aOther.y;
        z = z | aOther.z;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Or(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires Integral<T>
    {
        Vector3<T> ret;
        ret.x = aLeft.x | aRight.x;
        ret.y = aLeft.y | aRight.y;
        ret.z = aLeft.z | aRight.z;
        return ret;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE void Xor(const Vector3<T> &aOther)
        requires Integral<T>
    {
        x = x ^ aOther.x;
        y = y ^ aOther.y;
        z = z ^ aOther.z;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Xor(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires Integral<T>
    {
        Vector3<T> ret;
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
    DEVICE_CODE [[nodiscard]] static Vector3<T> Not(const Vector3<T> &aVec)
        requires Integral<T>
    {
        Vector3<T> ret;
        ret.x = ~aVec.x;
        ret.y = ~aVec.y;
        ret.z = ~aVec.z;
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
        y = std::exp(y);
        z = std::exp(z);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE void Exp()
        requires NonNativeType<T>
    {
        x = T::Exp(x);
        y = T::Exp(y);
        z = T::Exp(z);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE void Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = exp(x);
        y = exp(y);
        z = exp(z);
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Exp(const Vector3<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector3<T> ret;
        ret.x = exp(aVec.x);
        ret.y = exp(aVec.y);
        ret.z = exp(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Exp(const Vector3<T> &aVec)
        requires HostCode<T> && NativeType<T>
    {
        Vector3<T> ret;
        ret.x = std::exp(aVec.x);
        ret.y = std::exp(aVec.y);
        ret.z = std::exp(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Exp(const Vector3<T> &aVec)
        requires NonNativeType<T>
    {
        Vector3<T> ret;
        ret.x = T::Exp(aVec.x);
        ret.y = T::Exp(aVec.y);
        ret.z = T::Exp(aVec.z);
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
        y = std::log(y);
        z = std::log(z);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE void Ln()
        requires NonNativeType<T>
    {
        x = T::Ln(x);
        y = T::Ln(y);
        z = T::Ln(z);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE void Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = log(x);
        y = log(y);
        z = log(z);
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Ln(const Vector3<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector3<T> ret;
        ret.x = log(aVec.x);
        ret.y = log(aVec.y);
        ret.z = log(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Ln(const Vector3<T> &aVec)
        requires HostCode<T> && NativeType<T>
    {
        Vector3<T> ret;
        ret.x = std::log(aVec.x);
        ret.y = std::log(aVec.y);
        ret.z = std::log(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Ln(const Vector3<T> &aVec)
        requires NonNativeType<T>
    {
        Vector3<T> ret;
        ret.x = T::Ln(aVec.x);
        ret.y = T::Ln(aVec.y);
        ret.z = T::Ln(aVec.z);
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
        y = y * y;
        z = z * z;
    }

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Sqr(const Vector3<T> &aVec)
    {
        Vector3<T> ret;
        ret.x = aVec.x * aVec.x;
        ret.y = aVec.y * aVec.y;
        ret.z = aVec.z * aVec.z;
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
        y = std::sqrt(y);
        z = std::sqrt(z);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE void Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = sqrt(x);
        y = sqrt(y);
        z = sqrt(z);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE void Sqrt()
        requires NonNativeType<T>
    {
        x = T::Sqrt(x);
        y = T::Sqrt(y);
        z = T::Sqrt(z);
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Sqrt(const Vector3<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        Vector3<T> ret;
        ret.x = sqrt(aVec.x);
        ret.y = sqrt(aVec.y);
        ret.z = sqrt(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Sqrt(const Vector3<T> &aVec)
        requires HostCode<T> && NativeType<T>
    {
        Vector3<T> ret;
        ret.x = std::sqrt(aVec.x);
        ret.y = std::sqrt(aVec.y);
        ret.z = std::sqrt(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Sqrt(const Vector3<T> &aVec)
        requires NonNativeType<T>
    {
        Vector3<T> ret;
        ret.x = T::Sqrt(aVec.x);
        ret.y = T::Sqrt(aVec.y);
        ret.z = T::Sqrt(aVec.z);
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
        y = std::abs(y);
        z = std::abs(z);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE void Abs()
        requires SignedNumber<T> && NonNativeType<T>
    {
        x = T::Abs(x);
        y = T::Abs(y);
        z = T::Abs(z);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE void Abs()
        requires DeviceCode<T> && SignedNumber<T> && NativeType<T>
    {
        x = abs(x);
        y = abs(y);
        z = abs(z);
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Abs(const Vector3<T> &aVec)
        requires DeviceCode<T> && SignedNumber<T> && NativeType<T>
    {
        Vector3<T> ret;
        ret.x = abs(aVec.x);
        ret.y = abs(aVec.y);
        ret.z = abs(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Abs(const Vector3<T> &aVec)
        requires HostCode<T> && SignedNumber<T> && NativeType<T>
    {
        Vector3<T> ret;
        ret.x = std::abs(aVec.x);
        ret.y = std::abs(aVec.y);
        ret.z = std::abs(aVec.z);
        return ret;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Abs(const Vector3<T> &aVec)
        requires NonNativeType<T>
    {
        Vector3<T> ret;
        ret.x = T::Abs(aVec.x);
        ret.y = T::Abs(aVec.y);
        ret.z = T::Abs(aVec.z);
        return ret;
    }
#pragma endregion

#pragma region AbsDiff
    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector3<T> &aOther)
        requires HostCode<T> && NativeType<T>
    {
        x = std::abs(x - aOther.x);
        y = std::abs(y - aOther.y);
        z = std::abs(z - aOther.z);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> AbsDiff(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        Vector3<T> ret;
        ret.x = std::abs(aLeft.x - aRight.x);
        ret.y = std::abs(aLeft.y - aRight.y);
        ret.z = std::abs(aLeft.z - aRight.z);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector3<T> &aOther)
        requires DeviceCode<T> && NativeType<T>
    {
        x = abs(x - aOther.x);
        y = abs(y - aOther.y);
        z = abs(z - aOther.z);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> AbsDiff(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        Vector3<T> ret;
        ret.x = abs(aLeft.x - aRight.x);
        ret.y = abs(aLeft.y - aRight.y);
        ret.z = abs(aLeft.z - aRight.z);
        return ret;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE void AbsDiff(const Vector3<T> &aOther)
        requires NonNativeType<T>
    {
        x = T::Abs(x - aOther.x);
        y = T::Abs(y - aOther.y);
        z = T::Abs(z - aOther.z);
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> AbsDiff(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires NonNativeType<T>
    {
        Vector3<T> ret;
        ret.x = T::Abs(aLeft.x - aRight.x);
        ret.y = T::Abs(aLeft.y - aRight.y);
        ret.z = T::Abs(aLeft.z - aRight.z);
        return ret;
    }
#pragma endregion

#pragma region Dot
    /// <summary>
    /// Vector dot product
    /// </summary>
    DEVICE_CODE [[nodiscard]] static T Dot(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires FloatingPoint<T>
    {
        return aLeft.x * aRight.x + aLeft.y * aRight.y + aLeft.z * aRight.z;
    }

    /// <summary>
    /// Vector dot product
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Dot(const Vector3<T> &aRight) const
        requires FloatingPoint<T>
    {
        return x * aRight.x + y * aRight.y + z * aRight.z;
    }
#pragma endregion

#pragma region Cross
    /// <summary>
    /// Vector cross product
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Cross(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires FloatingPoint<T>
    {
        return Vector3<T>{aLeft.y * aRight.z - aLeft.z * aRight.y, aLeft.z * aRight.x - aLeft.x * aRight.z,
                          aLeft.x * aRight.y - aLeft.y * aRight.x};
    }

    /// <summary>
    /// Vector cross product
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<T> Cross(const Vector3<T> &aRight) const
        requires FloatingPoint<T>
    {
        return Vector3<T>{y * aRight.z - z * aRight.y, z * aRight.x - x * aRight.z, x * aRight.y - y * aRight.x};
    }
#pragma endregion

#pragma region Magnitude(Sqr)
    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        return sqrt(Dot(*this, *this));
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    [[nodiscard]] T Magnitude() const
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        return std::sqrt(Dot(*this, *this));
    }

    /// <summary>
    /// Vector length (L2 norm)
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Magnitude() const
        requires NonNativeType<T>
    {
        return T::Sqrt(Dot(*this, *this));
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
    DEVICE_CODE [[nodiscard]] static Vector3<T> Normalize(const Vector3<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector3<T> ret = aValue;
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
        y = max(aMinVal, min(y, aMaxVal));
        z = max(aMinVal, min(z, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    void Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeType<T>
    {
        x = std::max(aMinVal, std::min(x, aMaxVal));
        y = std::max(aMinVal, std::min(y, aMaxVal));
        z = std::max(aMinVal, std::min(z, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE void Clamp(T aMinVal, T aMaxVal)
        requires NonNativeType<T>
    {
        x = T::Max(aMinVal, T::Min(x, aMaxVal));
        y = T::Max(aMinVal, T::Min(y, aMaxVal));
        z = T::Max(aMinVal, T::Min(z, aMaxVal));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>) && (!IsHalfFp16<T> || !isSameType<TTarget, short>)
    {
        Clamp(T(numeric_limits<TTarget>::lowest()), T(numeric_limits<TTarget>::max()));
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <ComplexOrNumber TTarget>
    DEVICE_CODE void ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>) && IsHalfFp16<T> && isSameType<TTarget, short>
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
    DEVICE_CODE void Min(const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        x = min(x, aRight.x);
        y = min(y, aRight.y);
        z = min(z, aRight.z);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    void Min(const Vector3<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        x = std::min(x, aRight.x);
        y = std::min(y, aRight.y);
        z = std::min(z, aRight.z);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE void Min(const Vector3<T> &aRight)
        requires NonNativeType<T>
    {
        x.Min(aRight.x);
        y.Min(aRight.y);
        z.Min(aRight.z);
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Min(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        return Vector3<T>{T(min(aLeft.x, aRight.x)), T(min(aLeft.y, aRight.y)), T(min(aLeft.z, aRight.z))};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector3<T> Min(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        return Vector3<T>{std::min(aLeft.x, aRight.x), std::min(aLeft.y, aRight.y), std::min(aLeft.z, aRight.z)};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector3<T> Min(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires NonNativeType<T>
    {
        return Vector3<T>{T::Min(aLeft.x, aRight.x), T::Min(aLeft.y, aRight.y), T::Min(aLeft.z, aRight.z)};
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T> && NativeType<T>
    {
        return min(min(x, y), z);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T> && NonNativeType<T>
    {
        return T::Min(T::Min(x, y), z);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    [[nodiscard]] T Min() const
        requires HostCode<T>
    {
        return std::min({x, y, z});
    }
#pragma endregion

#pragma region Max
    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE void Max(const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        x = max(x, aRight.x);
        y = max(y, aRight.y);
        z = max(z, aRight.z);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    void Max(const Vector3<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        x = std::max(x, aRight.x);
        y = std::max(y, aRight.y);
        z = std::max(z, aRight.z);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE void Max(const Vector3<T> &aRight)
        requires NonNativeType<T>
    {
        x.Max(aRight.x);
        y.Max(aRight.y);
        z.Max(aRight.z);
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Max(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeType<T>
    {
        return Vector3<T>{T(max(aLeft.x, aRight.x)), T(max(aLeft.y, aRight.y)), T(max(aLeft.z, aRight.z))};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector3<T> Max(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires HostCode<T> && NativeType<T>
    {
        return Vector3<T>{std::max(aLeft.x, aRight.x), std::max(aLeft.y, aRight.y), std::max(aLeft.z, aRight.z)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Max(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires NonNativeType<T>
    {
        return Vector3<T>{T::Max(aLeft.x, aRight.x), T::Max(aLeft.y, aRight.y), T::Max(aLeft.z, aRight.z)};
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T> && NativeType<T>
    {
        return max(max(x, y), z);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T> && NonNativeType<T>
    {
        return T::Max(T::Max(x, y), z);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    [[nodiscard]] T Max() const
        requires HostCode<T>
    {
        return std::max({x, y, z});
    }
#pragma endregion

#pragma region Round
    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Round(const Vector3<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.Round();
        return ret;
    }

    /// <summary>
    /// Element wise round() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<int> RoundI(const Vector3<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.Round();
        return Vector3<int>(ret);
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
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = round(x);
        y = round(y);
        z = round(z);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    void Round()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::round(x);
        y = std::round(y);
        z = std::round(z);
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_ONLY_CODE void Round()
        requires NonNativeType<T>
    {
        x.Round();
        y.Round();
        z.Round();
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Floor(const Vector3<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.Floor();
        return ret;
    }

    /// <summary>
    /// Element wise floor() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<int> FloorI(const Vector3<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.Floor();
        return Vector3<int>(ret);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE void Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = floor(x);
        y = floor(y);
        z = floor(z);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    void Floor()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::floor(x);
        y = std::floor(y);
        z = std::floor(z);
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE void Floor()
        requires NonNativeType<T>
    {
        x.Floor();
        y.Floor();
        z.Floor();
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Ceil(const Vector3<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.Ceil();
        return ret;
    }

    /// <summary>
    /// Element wise ceil() and return as integer
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<int> CeilI(const Vector3<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.Ceil();
        return Vector3<int>(ret);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE void Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = ceil(x);
        y = ceil(y);
        z = ceil(z);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    void Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::ceil(x);
        y = std::ceil(y);
        z = std::ceil(z);
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE void Ceil()
        requires NonNativeType<T>
    {
        x.Ceil();
        y.Ceil();
        z.Ceil();
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RoundNearest(const Vector3<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector3<T> ret = aValue;
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
        y = __float2int_rn(y);
        z = __float2int_rn(z);
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    void RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::nearbyint(x);
        y = std::nearbyint(y);
        z = std::nearbyint(z);
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE void RoundNearest()
        requires NonNativeType<T>
    {
        x.RoundNearest();
        y.RoundNearest();
        z.RoundNearest();
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RoundZero(const Vector3<T> &aValue)
        requires FloatingPoint<T>
    {
        Vector3<T> ret = aValue;
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
        y = __float2int_rz(y);
        z = __float2int_rz(z);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    void RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::trunc(x);
        y = std::trunc(y);
        z = std::trunc(z);
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE void RoundZero()
        requires NonNativeType<T>
    {
        x.RoundZero();
        y.RoundZero();
        z.RoundZero();
    }
#pragma endregion

#pragma region Compare per element
    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareEQ(const Vector3<T> &aLeft, const Vector3<T> &aRight)
    {
        Vector3<byte> ret;
        ret.x = byte(aLeft.x == aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y == aRight.y) * TRUE_VALUE;
        ret.z = byte(aLeft.z == aRight.z) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareGE(const Vector3<T> &aLeft, const Vector3<T> &aRight)
    {
        Vector3<byte> ret;
        ret.x = byte(aLeft.x >= aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y >= aRight.y) * TRUE_VALUE;
        ret.z = byte(aLeft.z >= aRight.z) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareGT(const Vector3<T> &aLeft, const Vector3<T> &aRight)
    {
        Vector3<byte> ret;
        ret.x = byte(aLeft.x > aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y > aRight.y) * TRUE_VALUE;
        ret.z = byte(aLeft.z > aRight.z) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareLE(const Vector3<T> &aLeft, const Vector3<T> &aRight)
    {
        Vector3<byte> ret;
        ret.x = byte(aLeft.x <= aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y <= aRight.y) * TRUE_VALUE;
        ret.z = byte(aLeft.z <= aRight.z) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareLT(const Vector3<T> &aLeft, const Vector3<T> &aRight)
    {
        Vector3<byte> ret;
        ret.x = byte(aLeft.x < aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y < aRight.y) * TRUE_VALUE;
        ret.z = byte(aLeft.z < aRight.z) * TRUE_VALUE;
        return ret;
    }

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareNEQ(const Vector3<T> &aLeft, const Vector3<T> &aRight)
    {
        Vector3<byte> ret;
        ret.x = byte(aLeft.x != aRight.x) * TRUE_VALUE;
        ret.y = byte(aLeft.y != aRight.y) * TRUE_VALUE;
        ret.z = byte(aLeft.z != aRight.z) * TRUE_VALUE;
        return ret;
    }
#pragma endregion

#pragma region Data accessors
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
#pragma endregion
};

template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator+(const Vector3<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector3<T>{T(aLeft.x + aRight), T(aLeft.y + aRight), T(aLeft.z + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator+(T2 aLeft, const Vector3<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector3<T>{T(aLeft + aRight.x), T(aLeft + aRight.y), T(aLeft + aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator-(const Vector3<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector3<T>{T(aLeft.x - aRight), T(aLeft.y - aRight), T(aLeft.z - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator-(T2 aLeft, const Vector3<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector3<T>{T(aLeft - aRight.x), T(aLeft - aRight.y), T(aLeft - aRight.z)};
}

template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator*(const Vector3<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector3<T>{T(aLeft.x * aRight), T(aLeft.y * aRight), T(aLeft.z * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator*(T2 aLeft, const Vector3<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector3<T>{T(aLeft * aRight.x), T(aLeft * aRight.y), T(aLeft * aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator/(const Vector3<T> &aLeft, T2 aRight)
    requires ComplexOrNumber<T2>
{
    return Vector3<T>{T(aLeft.x / aRight), T(aLeft.y / aRight), T(aLeft.z / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator/(T2 aLeft, const Vector3<T> &aRight)
    requires ComplexOrNumber<T2>
{
    return Vector3<T>{T(aLeft / aRight.x), T(aLeft / aRight.y), T(aLeft / aRight.z)};
}

template <HostCode T2> std::ostream &operator<<(std::ostream &aOs, const Vector3<T2> &aVec)
{
    aOs << '(' << aVec.x << ", " << aVec.y << ", " << aVec.z << ')';
    return aOs;
}

template <HostCode T2> std::wostream &operator<<(std::wostream &aOs, const Vector3<T2> &aVec)
{
    aOs << '(' << aVec.x << ", " << aVec.y << ", " << aVec.z << ')';
    return aOs;
}

template <HostCode T2> std::istream &operator>>(std::istream &aIs, Vector3<T2> &aVec)
{
    aIs >> aVec.x >> aVec.y >> aVec.z;
    return aIs;
}

template <HostCode T2> std::wistream &operator>>(std::wistream &aIs, Vector3<T2> &aVec)
{
    aIs >> aVec.x >> aVec.y >> aVec.z;
    return aIs;
}

} // namespace opp
