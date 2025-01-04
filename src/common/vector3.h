#pragma once
#include "defines.h"
#include "exception.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "numeric_limits.h"
#include "safeCast.h"
#include "vector_typetraits.h"
#include <cmath>
#include <common/utilities.h>
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
template <Number T> struct alignas(sizeof(T)) Vector3
{
    T x;
    T y;
    T z;

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    Vector3() noexcept = default;

    /// <summary>
    /// Initializes vector to all components = aVal
    /// </summary>
    DEVICE_CODE Vector3(T aVal) noexcept : x(aVal), y(aVal), z(aVal)
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal (especially when set to 0)
    /// </summary>
    DEVICE_CODE Vector3(int aVal) noexcept
        requires(!IsInt<T>)
        : x(static_cast<T>(aVal)), y(static_cast<T>(aVal)), z(static_cast<T>(aVal))
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0], aVal[1], aVal[2]]
    /// </summary>
    DEVICE_CODE explicit Vector3(T aVal[3]) noexcept : x(aVal[0]), y(aVal[1]), z(aVal[2])
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
    template <Number T2> DEVICE_CODE Vector3(const Vector3<T2> &aVec) noexcept
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
    template <Number T2> DEVICE_CODE Vector3(Vector3<T2> &aVec) noexcept
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
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Vector3 aLeft, Vector3 aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);
        MakeNANandINFValid(aLeft.z, aRight.z);

        bool res = std::abs(aLeft.x - aRight.x) <= aEpsilon;
        res &= std::abs(aLeft.y - aRight.y) <= aEpsilon;
        res &= std::abs(aLeft.z - aRight.z) <= aEpsilon;
        return res;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Vector3 aLeft, Vector3 aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);
        MakeNANandINFValid(aLeft.z, aRight.z);

        bool res = abs(aLeft.x - aRight.x) <= aEpsilon;
        res &= abs(aLeft.y - aRight.y) <= aEpsilon;
        res &= abs(aLeft.z - aRight.z) <= aEpsilon;
        return res;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(Vector3 aLeft, Vector3 aRight, T aEpsilon)
        requires Is16BitFloat<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);
        MakeNANandINFValid(aLeft.z, aRight.z);

        bool res = T::Abs(aLeft.x - aRight.x) <= aEpsilon;
        res &= T::Abs(aLeft.y - aRight.y) <= aEpsilon;
        res &= T::Abs(aLeft.z - aRight.z) <= aEpsilon;
        return res;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector3 &aLeft, const Vector3 &aRight,
                                                complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>
    {
        bool res = T::EqEps(aLeft.x, aRight.x, aEpsilon);
        res &= T::EqEps(aLeft.y, aRight.y, aEpsilon);
        res &= T::EqEps(aLeft.z, aRight.z, aEpsilon);
        return res;
    }

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(const Vector3 &aOther) const
        requires RealNumber<T>
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
        requires RealNumber<T>
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
        requires RealNumber<T>
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
        requires RealNumber<T>
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
        requires RealSignedNumber<T> || ComplexNumber<T>
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
        return x;
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
    template <Number T2> [[nodiscard]] static Vector3<T> DEVICE_CODE Convert(const Vector3<T2> &aVec)
    {
        return {static_cast<T>(aVec.x), static_cast<T>(aVec.y), static_cast<T>(aVec.z)};
    }
#pragma endregion

#pragma region Integral only Methods
    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector3<T> &LShift(const Vector3<T> &aOther)
        requires RealIntegral<T>
    {
        x = x << aOther.x;
        y = y << aOther.y;
        z = z << aOther.z;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> LShift(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>
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
    DEVICE_CODE Vector3<T> &RShift(const Vector3<T> &aOther)
        requires RealIntegral<T>
    {
        x = x >> aOther.x;
        y = y >> aOther.y;
        z = z >> aOther.z;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RShift(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>
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
    DEVICE_CODE Vector3<T> &LShift(const T &aOther)
        requires RealIntegral<T>
    {
        x = x << aOther;
        y = y << aOther;
        z = z << aOther;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> LShift(const Vector3<T> &aLeft, const T &aRight)
        requires RealIntegral<T>
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
    DEVICE_CODE Vector3<T> &RShift(const T &aOther)
        requires RealIntegral<T>
    {
        x = x >> aOther;
        y = y >> aOther;
        z = z >> aOther;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RShift(const Vector3<T> &aLeft, const T &aRight)
        requires RealIntegral<T>
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
    DEVICE_CODE Vector3<T> &And(const Vector3<T> &aOther)
        requires RealIntegral<T>
    {
        x = x & aOther.x;
        y = y & aOther.y;
        z = z & aOther.z;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> And(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>
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
    DEVICE_CODE Vector3<T> &Or(const Vector3<T> &aOther)
        requires RealIntegral<T>
    {
        x = x | aOther.x;
        y = y | aOther.y;
        z = z | aOther.z;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Or(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>
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
    DEVICE_CODE Vector3<T> &Xor(const Vector3<T> &aOther)
        requires RealIntegral<T>
    {
        x = x ^ aOther.x;
        y = y ^ aOther.y;
        z = z ^ aOther.z;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Xor(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>
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
    DEVICE_CODE Vector3<T> &Not()
        requires RealIntegral<T>
    {
        x = ~x;
        y = ~y;
        z = ~z;
        return *this;
    }

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Not(const Vector3<T> &aVec)
        requires RealIntegral<T>
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
    Vector3<T> &Exp()
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::exp(x);
        y = std::exp(y);
        z = std::exp(z);
        return *this;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector3<T> &Exp()
        requires NonNativeNumber<T>
    {
        x = T::Exp(x);
        y = T::Exp(y);
        z = T::Exp(z);
        return *this;
    }

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector3<T> &Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = exp(x);
        y = exp(y);
        z = exp(z);
        return *this;
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
        requires HostCode<T> && NativeNumber<T>
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
        requires NonNativeNumber<T>
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
    Vector3<T> &Ln()
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::log(x);
        y = std::log(y);
        z = std::log(z);
        return *this;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector3<T> &Ln()
        requires NonNativeNumber<T>
    {
        x = T::Ln(x);
        y = T::Ln(y);
        z = T::Ln(z);
        return *this;
    }

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector3<T> &Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = log(x);
        y = log(y);
        z = log(z);
        return *this;
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
        requires HostCode<T> && NativeNumber<T>
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
        requires NonNativeNumber<T>
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
    DEVICE_CODE Vector3<T> &Sqr()
    {
        x = x * x;
        y = y * y;
        z = z * z;
        return *this;
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
    Vector3<T> &Sqrt()
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::sqrt(x);
        y = std::sqrt(y);
        z = std::sqrt(z);
        return *this;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector3<T> &Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = sqrt(x);
        y = sqrt(y);
        z = sqrt(z);
        return *this;
    }

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector3<T> &Sqrt()
        requires NonNativeNumber<T>
    {
        x = T::Sqrt(x);
        y = T::Sqrt(y);
        z = T::Sqrt(z);
        return *this;
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
        requires HostCode<T> && NativeNumber<T>
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
        requires NonNativeNumber<T>
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
    Vector3<T> &Abs()
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        x = std::abs(x);
        y = std::abs(y);
        z = std::abs(z);
        return *this;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector3<T> &Abs()
        requires RealSignedNumber<T> && NonNativeNumber<T>
    {
        x = T::Abs(x);
        y = T::Abs(y);
        z = T::Abs(z);
        return *this;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector3<T> &Abs()
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
    {
        x = abs(x);
        y = abs(y);
        z = abs(z);
        return *this;
    }

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Abs(const Vector3<T> &aVec)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
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
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
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
        requires RealSignedNumber<T> && NonNativeNumber<T>
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
    DEVICE_CODE Vector3<T> &AbsDiff(const Vector3<T> &aOther)
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::abs(x - aOther.x);
        y = std::abs(y - aOther.y);
        z = std::abs(z - aOther.z);
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> AbsDiff(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires HostCode<T> && NativeFloatingPoint<T>
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
    DEVICE_CODE Vector3<T> &AbsDiff(const Vector3<T> &aOther)
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = abs(x - aOther.x);
        y = abs(y - aOther.y);
        z = abs(z - aOther.z);
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> AbsDiff(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeFloatingPoint<T>
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
    DEVICE_CODE Vector3<T> &AbsDiff(const Vector3<T> &aOther)
        requires RealSignedNumber<T> && NonNativeNumber<T>
    {
        x = T::Abs(x - aOther.x);
        y = T::Abs(y - aOther.y);
        z = T::Abs(z - aOther.z);
        return *this;
    }

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> AbsDiff(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealSignedNumber<T> && NonNativeNumber<T>
    {
        Vector3<T> ret;
        ret.x = T::Abs(aLeft.x - aRight.x);
        ret.y = T::Abs(aLeft.y - aRight.y);
        ret.z = T::Abs(aLeft.z - aRight.z);
        return ret;
    }
#pragma endregion

#pragma region Methods for Complex types
    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE Vector3<T> &Conj()
        requires ComplexNumber<T>
    {
        x.Conj();
        y.Conj();
        z.Conj();
        return *this;
    }

    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Conj(const Vector3<T> &aValue)
        requires ComplexNumber<T>
    {
        return {T::Conj(aValue.x), T::Conj(aValue.y), T::Conj(aValue.z)};
    }

    /// <summary>
    /// Conjugate complex multiplication: this * conj(aOther)  per element
    /// </summary>
    DEVICE_CODE Vector3<T> &ConjMul(const Vector3<T> &aOther)
        requires ComplexNumber<T>
    {
        x.ConjMul(aOther.x);
        y.ConjMul(aOther.y);
        z.ConjMul(aOther.z);
        return *this;
    }

    /// <summary>
    /// Conjugate complex multiplication: aLeft * conj(aRight) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> ConjMul(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires ComplexNumber<T>
    {
        return {T::ConjMul(aLeft.x, aRight.x), T::ConjMul(aLeft.y, aRight.y), T::ConjMul(aLeft.z, aRight.z)};
    }

    /// <summary>
    /// Complex magnitude per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<complex_basetype_t<T>> Magnitude() const
        requires ComplexFloatingPoint<T>
    {
        Vector3<complex_basetype_t<T>> ret;
        ret.x = x.Magnitude();
        ret.y = y.Magnitude();
        ret.z = z.Magnitude();
        return ret;
    }

    /// <summary>
    /// Complex magnitude squared per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<complex_basetype_t<T>> MagnitudeSqr() const
        requires ComplexFloatingPoint<T>
    {
        Vector3<complex_basetype_t<T>> ret;
        ret.x = x.MagnitudeSqr();
        ret.y = y.MagnitudeSqr();
        ret.z = z.MagnitudeSqr();
        return ret;
    }

    /// <summary>
    /// Angle between real and imaginary of a complex number (atan2(image, real)) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<complex_basetype_t<T>> Angle() const
        requires ComplexFloatingPoint<T>
    {
        Vector3<complex_basetype_t<T>> ret;
        ret.x = x.Angle();
        ret.y = y.Angle();
        ret.z = z.Angle();
        return ret;
    }
#pragma endregion

#pragma region Clamp
    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector3<T> &Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeNumber<T>
    {
        x = max(aMinVal, min(x, aMaxVal));
        y = max(aMinVal, min(y, aMaxVal));
        z = max(aMinVal, min(z, aMaxVal));
        return *this;
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    Vector3<T> &Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::max(aMinVal, std::min(x, aMaxVal));
        y = std::max(aMinVal, std::min(y, aMaxVal));
        z = std::max(aMinVal, std::min(z, aMaxVal));
        return *this;
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector3<T> &Clamp(T aMinVal, T aMaxVal)
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        x = T::Max(aMinVal, T::Min(x, aMaxVal));
        y = T::Max(aMinVal, T::Min(y, aMaxVal));
        z = T::Max(aMinVal, T::Min(z, aMaxVal));
        return *this;
    }

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector3<T> &Clamp(complex_basetype_t<T> aMinVal, complex_basetype_t<T> aMaxVal)
        requires ComplexNumber<T>
    {
        x.Clamp(aMinVal, aMaxVal);
        y.Clamp(aMinVal, aMaxVal);
        z.Clamp(aMinVal, aMaxVal);
        return *this;
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector3<T> &ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>)
    {
        return Clamp(numeric_limits_conversion<T, TTarget>::lowest(), numeric_limits_conversion<T, TTarget>::max());
    }

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector3<T> &ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>)
    {
        return *this;
    }
#pragma endregion

#pragma region Min
    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector3<T> &Min(const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        x = min(x, aRight.x);
        y = min(y, aRight.y);
        z = min(z, aRight.z);
        return *this;
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    Vector3<T> &Min(const Vector3<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::min(x, aRight.x);
        y = std::min(y, aRight.y);
        z = std::min(z, aRight.z);
        return *this;
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector3<T> &Min(const Vector3<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        x.Min(aRight.x);
        y.Min(aRight.y);
        z.Min(aRight.z);
        return *this;
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Min(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Vector3<T>{T(min(aLeft.x, aRight.x)), T(min(aLeft.y, aRight.y)), T(min(aLeft.z, aRight.z))};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector3<T> Min(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        return Vector3<T>{std::min(aLeft.x, aRight.x), std::min(aLeft.y, aRight.y), std::min(aLeft.z, aRight.z)};
    }

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector3<T> Min(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        return Vector3<T>{T::Min(aLeft.x, aRight.x), T::Min(aLeft.y, aRight.y), T::Min(aLeft.z, aRight.z)};
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T> && NativeNumber<T>
    {
        return min(min(x, y), z);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        return T::Min(T::Min(x, y), z);
    }

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    [[nodiscard]] T Min() const
        requires HostCode<T> && NativeNumber<T>
    {
        return std::min({x, y, z});
    }
#pragma endregion

#pragma region Max
    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector3<T> &Max(const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        x = max(x, aRight.x);
        y = max(y, aRight.y);
        z = max(z, aRight.z);
        return *this;
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    Vector3<T> &Max(const Vector3<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        x = std::max(x, aRight.x);
        y = std::max(y, aRight.y);
        z = std::max(z, aRight.z);
        return *this;
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector3<T> &Max(const Vector3<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        x.Max(aRight.x);
        y.Max(aRight.y);
        z.Max(aRight.z);
        return *this;
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Max(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>
    {
        return Vector3<T>{T(max(aLeft.x, aRight.x)), T(max(aLeft.y, aRight.y)), T(max(aLeft.z, aRight.z))};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector3<T> Max(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires HostCode<T> && NativeNumber<T>
    {
        return Vector3<T>{std::max(aLeft.x, aRight.x), std::max(aLeft.y, aRight.y), std::max(aLeft.z, aRight.z)};
    }

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Max(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        return Vector3<T>{T::Max(aLeft.x, aRight.x), T::Max(aLeft.y, aRight.y), T::Max(aLeft.z, aRight.z)};
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T> && NativeNumber<T>
    {
        return max(max(x, y), z);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires NonNativeNumber<T> && (!ComplexNumber<T>)
    {
        return T::Max(T::Max(x, y), z);
    }

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    [[nodiscard]] T Max() const
        requires HostCode<T> && NativeNumber<T>
    {
        return std::max({x, y, z});
    }
#pragma endregion

#pragma region Round
    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Round(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.Round();
        return ret;
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Vector3<T> &Round()
        requires NonNativeFloatingPoint<T>
    {
        x.Round();
        y.Round();
        z.Round();
        return *this;
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Vector3<T> &Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = round(x);
        y = round(y);
        z = round(z);
        return *this;
    }

    /// <summary>
    /// Element wise round()
    /// </summary>
    Vector3<T> &Round()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::round(x);
        y = std::round(y);
        z = std::round(z);
        return *this;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Floor(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.Floor();
        return ret;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE Vector3<T> &Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = floor(x);
        y = floor(y);
        z = floor(z);
        return *this;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    Vector3<T> &Floor()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::floor(x);
        y = std::floor(y);
        z = std::floor(z);
        return *this;
    }

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE Vector3<T> &Floor()
        requires NonNativeFloatingPoint<T>
    {
        x.Floor();
        y.Floor();
        z.Floor();
        return *this;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Ceil(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.Ceil();
        return ret;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE Vector3<T> &Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = ceil(x);
        y = ceil(y);
        z = ceil(z);
        return *this;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    Vector3<T> &Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::ceil(x);
        y = std::ceil(y);
        z = std::ceil(z);
        return *this;
    }

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE Vector3<T> &Ceil()
        requires NonNativeFloatingPoint<T>
    {
        x.Ceil();
        y.Ceil();
        z.Ceil();
        return *this;
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RoundNearest(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.RoundNearest();
        return ret;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector3<T> &RoundNearest()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = rint(x);
        y = rint(y);
        z = rint(z);
        return *this;
    }

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    Vector3<T> &RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::nearbyint(x);
        y = std::nearbyint(y);
        z = std::nearbyint(z);
        return *this;
    }

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE Vector3<T> &RoundNearest()
        requires NonNativeFloatingPoint<T>
    {
        x.RoundNearest();
        y.RoundNearest();
        z.RoundNearest();
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RoundZero(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>
    {
        Vector3<T> ret = aValue;
        ret.RoundZero();
        return ret;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector3<T> &RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>
    {
        x = trunc(x);
        y = trunc(y);
        z = trunc(z);
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    Vector3<T> &RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>
    {
        x = std::trunc(x);
        y = std::trunc(y);
        z = std::trunc(z);
        return *this;
    }

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE Vector3<T> &RoundZero()
        requires NonNativeFloatingPoint<T>
    {
        x.RoundZero();
        y.RoundZero();
        z.RoundZero();
        return *this;
    }
#pragma endregion

#pragma region Compare per element
    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    [[nodiscard]] static Vector3<byte> CompareEQEps(Vector3<T> aLeft, Vector3<T> aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);
        MakeNANandINFValid(aLeft.z, aRight.z);

        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(std::abs(aLeft.x - aRight.x) <= aEpsilon) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(std::abs(aLeft.y - aRight.y) <= aEpsilon) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(std::abs(aLeft.z - aRight.z) <= aEpsilon) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareEQEps(Vector3<T> aLeft, Vector3<T> aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);
        MakeNANandINFValid(aLeft.z, aRight.z);

        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(abs(aLeft.x - aRight.x) <= aEpsilon) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(abs(aLeft.y - aRight.y) <= aEpsilon) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(abs(aLeft.z - aRight.z) <= aEpsilon) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareEQEps(Vector3<T> aLeft, Vector3<T> aRight, T aEpsilon)
        requires Is16BitFloat<T>
    {
        MakeNANandINFValid(aLeft.x, aRight.x);
        MakeNANandINFValid(aLeft.y, aRight.y);
        MakeNANandINFValid(aLeft.z, aRight.z);

        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(T::Abs(aLeft.x - aRight.x) <= aEpsilon) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(T::Abs(aLeft.y - aRight.y) <= aEpsilon) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(T::Abs(aLeft.z - aRight.z) <= aEpsilon) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareEQEps(const Vector3<T> &aLeft, const Vector3<T> &aRight,
                                                                complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>
    {
        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.x, aRight.x, aEpsilon)) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.y, aRight.y, aEpsilon)) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.z, aRight.z, aEpsilon)) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareEQ(const Vector3<T> &aLeft, const Vector3<T> &aRight)
    {
        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x == aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y == aRight.y) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(aLeft.z == aRight.z) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareGE(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealNumber<T>
    {
        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x >= aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y >= aRight.y) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(aLeft.z >= aRight.z) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareGT(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealNumber<T>
    {
        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x > aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y > aRight.y) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(aLeft.z > aRight.z) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareLE(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealNumber<T>
    {
        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x <= aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y <= aRight.y) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(aLeft.z <= aRight.z) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareLT(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealNumber<T>
    {
        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x < aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y < aRight.y) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(aLeft.z < aRight.z) * TRUE_VALUE);
        return ret;
    }

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareNEQ(const Vector3<T> &aLeft, const Vector3<T> &aRight)
    {
        Vector3<byte> ret;
        ret.x = static_cast<byte>(static_cast<int>(aLeft.x != aRight.x) * TRUE_VALUE);
        ret.y = static_cast<byte>(static_cast<int>(aLeft.y != aRight.y) * TRUE_VALUE);
        ret.z = static_cast<byte>(static_cast<int>(aLeft.z != aRight.z) * TRUE_VALUE);
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
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft.x + aRight), static_cast<T>(aLeft.y + aRight),
                      static_cast<T>(aLeft.z + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator+(T2 aLeft, const Vector3<T> &aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft + aRight.x), static_cast<T>(aLeft + aRight.y),
                      static_cast<T>(aLeft + aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator-(const Vector3<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft.x - aRight), static_cast<T>(aLeft.y - aRight),
                      static_cast<T>(aLeft.z - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator-(T2 aLeft, const Vector3<T> &aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft - aRight.x), static_cast<T>(aLeft - aRight.y),
                      static_cast<T>(aLeft - aRight.z)};
}

template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator*(const Vector3<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft.x * aRight), static_cast<T>(aLeft.y * aRight),
                      static_cast<T>(aLeft.z * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator*(T2 aLeft, const Vector3<T> &aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft * aRight.x), static_cast<T>(aLeft * aRight.y),
                      static_cast<T>(aLeft * aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator/(const Vector3<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft.x / aRight), static_cast<T>(aLeft.y / aRight),
                      static_cast<T>(aLeft.z / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator/(T2 aLeft, const Vector3<T> &aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft / aRight.x), static_cast<T>(aLeft / aRight.y),
                      static_cast<T>(aLeft / aRight.z)};
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

// byte and sbyte are treated as characters and not numbers:
template <HostCode T2>
std::ostream &operator<<(std::ostream &aOs, const Vector3<T2> &aVec)
    requires ByteSizeType<T2>
{
    aOs << '(' << static_cast<int>(aVec.x) << ", " << static_cast<int>(aVec.y) << ", " << static_cast<int>(aVec.z)
        << ')';
    return aOs;
}

template <HostCode T2>
std::wostream &operator<<(std::wostream &aOs, const Vector3<T2> &aVec)
    requires ByteSizeType<T2>
{
    aOs << '(' << static_cast<int>(aVec.x) << ", " << static_cast<int>(aVec.y) << ", " << static_cast<int>(aVec.z)
        << ')';
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

template <HostCode T2>
std::istream &operator>>(std::istream &aIs, Vector3<T2> &aVec)
    requires ByteSizeType<T2>
{
    int temp = 0;
    aIs >> temp;
    aVec.x = static_cast<T2>(temp);
    aIs >> temp;
    aVec.y = static_cast<T2>(temp);
    aIs >> temp;
    aVec.z = static_cast<T2>(temp);
    return aIs;
}

template <HostCode T2>
std::wistream &operator>>(std::wistream &aIs, Vector3<T2> &aVec)
    requires ByteSizeType<T2>
{
    int temp = 0;
    aIs >> temp;
    aVec.x = static_cast<T2>(temp);
    aIs >> temp;
    aVec.y = static_cast<T2>(temp);
    aIs >> temp;
    aVec.z = static_cast<T2>(temp);
    return aIs;
}

} // namespace opp
