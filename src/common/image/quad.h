#pragma once
#include <common/defines.h>
#include <common/image/roi.h>
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
/// A Quad represents the coordinates of the corners of an image ROI.<para/>
/// For the un-transformed original ROI the four coordinates are <para/>
/// p0 = (X, Y)<para/>
/// p1 = (X + width - 1, Y)<para/>
/// p2 = (X + width - 1, Y + height - 1)<para/>
/// p3 = (X, Y + height - 1)<para/>
/// With transformed coordinates, the points coorespond to the same ROI corners.
/// </summary>
template <RealFloatingPoint T> struct Quad
{
    Vector2<T> P0;
    Vector2<T> P1;
    Vector2<T> P2;
    Vector2<T> P3;

    Quad() : P0(0), P1(0), P2(0), P3(0)
    {
    }

    Quad(const Vector2<T> &aP0, const Vector2<T> &aP1, const Vector2<T> &aP2, const Vector2<T> &aP3)
        : P0(aP0), P1(aP1), P2(aP2), P3(aP3)
    {
    }

    Quad(const Roi &aRoi)
        : P0(aRoi.FirstPixel()), P1(aRoi.LastX(), aRoi.FirstY()), P2(aRoi.LastPixel()), P3(aRoi.FirstX(), aRoi.LastY())
    {
    }

    friend std::ostream &operator<<(std::ostream &aOs, const Quad<T> &aQuad)
    {
        std::streamsize maxSize = 0;

        maxSize = std::max(std::streamsize(std::to_string(aQuad.P0.x).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P0.y).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P1.x).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P1.y).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P2.x).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P2.y).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P3.x).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P3.y).length()), maxSize);

        // clang tidy gets a bit crazy with std::streamsize...
        // NOLINTBEGIN
        aOs << '(' << std::setw(maxSize) << std::to_string(aQuad.P0.x) << " " << std::setw(maxSize)
            << std::to_string(aQuad.P0.y) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aQuad.P1.x) << " " << std::setw(maxSize)
            << std::to_string(aQuad.P1.y) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aQuad.P2.x) << " " << std::setw(maxSize)
            << std::to_string(aQuad.P2.y) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aQuad.P3.x) << " " << std::setw(maxSize)
            << std::to_string(aQuad.P3.y) << ')' << std::endl;
        return aOs;
        // NOLINTEND
    }
    friend std::wostream &operator<<(std::wostream &aOs, const Quad<T> &aQuad)
    {
        std::streamsize maxSize = 0;

        maxSize = std::max(std::streamsize(std::to_string(aQuad.P0.x).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P0.y).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P1.x).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P1.y).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P2.x).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P2.y).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P3.x).length()), maxSize);
        maxSize = std::max(std::streamsize(std::to_string(aQuad.P3.y).length()), maxSize);

        // clang tidy gets a bit crazy with std::streamsize...
        // NOLINTBEGIN
        aOs << '(' << std::setw(maxSize) << std::to_string(aQuad.P0.x) << " " << std::setw(maxSize)
            << std::to_string(aQuad.P0.y) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aQuad.P1.x) << " " << std::setw(maxSize)
            << std::to_string(aQuad.P1.y) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aQuad.P2.x) << " " << std::setw(maxSize)
            << std::to_string(aQuad.P2.y) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aQuad.P3.x) << " " << std::setw(maxSize)
            << std::to_string(aQuad.P3.y) << ')' << std::endl;
        return aOs;
        // NOLINTEND
    }
    friend std::istream &operator>>(std::istream &aIs, Quad<T> &aQuad)
    {
        aIs >> aQuad.P0 >> aQuad.P1 >> aQuad.P2 >> aQuad.P3;
        return aIs;
    }
    friend std::wistream &operator>>(std::wistream &aIs, Quad<T> &aQuad)
    {
        aIs >> aQuad.P0 >> aQuad.P1 >> aQuad.P2 >> aQuad.P3;
        return aIs;
    }

    bool operator==(const Quad &aOther) const
    {
        return aOther.P0 == P0 && aOther.P1 == P1 && aOther.P2 == P2 && aOther.P3 == P3;
    }
    bool operator!=(const Quad &aOther) const
    {
        return aOther.P0 != P0 || aOther.P1 != P1 || aOther.P2 != P2 || aOther.P3 != P3;
    }
};
} // namespace mpp::image