#pragma once
#include <common/defines.h>
#include <common/image/quad.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <iomanip>
#include <ios>
#include <istream>
#include <ostream>
#include <string>
#include <utility>

namespace mpp::image
{
/// <summary>
/// A Bound represents the minimum and maximum extent of a Quad.
/// </summary>
template <RealFloatingPoint T> struct Bound
{
    Vector2<T> Min;
    Vector2<T> Max;

    Bound()
    {
    }

    Bound(const Vector2<T> &aMin, const Vector2<T> &aMax) : Min(aMin), Max(aMax)
    {
    }

    Bound(const Quad<T> &Quad)
        : Min(Vector2<T>::Min(Vector2<T>::Min(Quad.P0, Quad.P1), Vector2<T>::Min(Quad.P2, Quad.P3))),
          Max(Vector2<T>::Max(Vector2<T>::Max(Quad.P0, Quad.P1), Vector2<T>::Max(Quad.P2, Quad.P3)))
    {
    }

    friend std::ostream &operator<<(std::ostream &aOs, const Bound<T> &aBound)
    {
        aOs << "Min: " << aBound.Min << " Max: " << aBound.Max;
        return aOs;
    }
    friend std::wostream &operator<<(std::wostream &aOs, const Bound<T> &aBound)
    {
        aOs << "Min: " << aBound.Min << " Max: " << aBound.Max;
        return aOs;
    }
    friend std::istream &operator>>(std::istream &aIs, Bound<T> &aBound)
    {
        aIs >> aBound.Min >> aBound.Max;
        return aIs;
    }
    friend std::wistream &operator>>(std::wistream &aIs, Bound<T> &aBound)
    {
        aIs >> aBound.Min >> aBound.Max;
        return aIs;
    }

    bool operator==(const Bound &aOther) const
    {
        return aOther.Min == Min && aOther.Max == Max;
    }
    bool operator!=(const Bound &aOther) const
    {
        return aOther.Min != Min || aOther.Max != Max;
    }
};
} // namespace mpp::image