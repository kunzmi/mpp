#pragma once
#include <common/defines.h>
#include <common/image/border.h>
#include <common/image/size2D.h>
#include <common/vectorTypes.h>
#include <istream>
#include <ostream>

namespace opp::image
{
/// <summary>
/// A 2D roi represented as a start pixel and a size
/// Note: we avoid any absolute referencing like top/left or lower/right as an image can have any orientation. Instead
/// we only reference lower X -> higher X according to the coordinate value.
/// Note: in device code, only the value members are available.
/// </summary>
struct alignas(4 * sizeof(int)) Roi
{
    int x{0};      // NOLINT(misc-non-private-member-variables-in-classes)
    int y{0};      // NOLINT(misc-non-private-member-variables-in-classes)
    int width{0};  // NOLINT(misc-non-private-member-variables-in-classes)
    int height{0}; // NOLINT(misc-non-private-member-variables-in-classes)

    /// <summary>
    /// Initializes a new roi to all components = 0
    /// </summary>
    Roi() noexcept = default;

    /// <summary>
    /// Initializes a new roi to (aX, aY, aWidth, aHeight)
    /// </summary>
    Roi(int aX, int aY, int aWidth, int aHeight) noexcept;

    /// <summary>
    /// Initializes a new roi to (aX, aY, aSize.x, aSize.y)
    /// </summary>
    Roi(int aX, int aY, const Size2D &aSize) noexcept;

    /// <summary>
    /// Initializes a new roi to (aPos.x, aPos.y, aSize.x, aSize.y)
    /// </summary>
    Roi(const Vec2i &aPos, const Size2D &aSize) noexcept;

    /// <summary>
    /// Initializes a new roi to (aPos.x, aPos.y, aWidth, aHeight)
    /// </summary>
    Roi(const Vec2i &aPos, int aWidth, int aHeight) noexcept;

    /// <summary>
    /// Initializes a new roi to (aVec.x, aVec.y, aVec.z, aVec.w)
    /// </summary>
    Roi(const Vec4i &aVec) noexcept; // NOLINT(hicpp-explicit-conversions)

    /// <summary>
    /// Initializes a new roi to (aArr[0], aArr[1], aArr[2], aArr[3])
    /// </summary>
    Roi(int aArr[4]) noexcept; // NOLINT(hicpp-explicit-conversions)

    ~Roi() = default;

    Roi(const Roi &)                     = default;
    Roi(Roi &&)                          = default;
    Roi &operator=(const Roi &) noexcept = default;
    Roi &operator=(Roi &&) noexcept      = default;

    /// <summary>
    /// Gets the lowest x-coordinate.
    /// </summary>
    [[nodiscard]] int FirstX() const;

    /// <summary>
    /// Gets the lowest y-coordinate.
    /// </summary>
    [[nodiscard]] int FirstY() const;

    /// <summary>
    /// Gets the y-coordinate that is the sum of the y and height values - 1.
    /// </summary>
    [[nodiscard]] int LastY() const;

    /// <summary>
    /// Gets the x-coordinate that is the sum of x and width values - 1.
    /// </summary>
    [[nodiscard]] int LastX() const;

    /// <summary>
    /// Gets the first x/y-coordinate inside the roi.
    /// </summary>
    [[nodiscard]] Vec2i FirstPixel() const;

    /// <summary>
    /// Gets the x/y-coordinate of the last pixel inside the roi.
    /// </summary>
    [[nodiscard]] Vec2i LastPixel() const;

    /// <summary>
    /// Gets the x/y-coordinate of the first pixel inside the roi.
    /// </summary>
    [[nodiscard]] Vec2i BoundingBoxMin() const;

    /// <summary>
    /// Gets the x/y-coordinate of first pixel outside the roi.
    /// </summary>
    [[nodiscard]] Vec2i BoundingBoxMax() const;

    /// <summary>
    /// Gets the width and height of the ROI.
    /// </summary>
    [[nodiscard]] Size2D Size() const;

    /// <summary>
    /// Enlarges the roi by the specified amount (add border)
    /// </summary>
    Roi &operator+=(int aOther);

    /// <summary>
    /// Enlarges the roi by the specified amount (add border), different values for X and Y
    /// </summary>
    Roi &operator+=(const Vec2i &aOther);

    /// <summary>
    /// Enlarges the roi by the specified amount (add border)
    /// </summary>
    Roi &operator+=(const Border &aOther);

    /// <summary>
    /// Enlarges the roi so that the new roi contains both source roi. (union)
    /// </summary>
    Roi &operator+=(const Roi &aOther);

    /// <summary>
    /// Reduces the roi by the specified amount (subtract border)
    /// </summary>
    Roi &operator-=(int aOther);

    /// <summary>
    /// Reduces the roi by the specified amount (subtract border)
    /// </summary>
    Roi &operator-=(const Border &aOther);

    /// <summary>
    /// Reduces the roi by the specified amount (subtract border), different values for X and Y
    /// </summary>
    Roi &operator-=(const Vec2i &aOther);

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    Roi &operator*=(int aOther);

    /// <summary>
    /// Component wise division
    /// </summary>
    Roi &operator/=(int aOther);

    /// <summary>
    /// returns true if all roi components are equal
    /// </summary>
    bool operator==(const Roi &aOther) const;

    /// <summary>
    /// returns true if at least one roi component is not equal
    /// </summary>
    bool operator!=(const Roi &aOther) const;

    /// <summary>
    /// returns true if the specified point is within the roi area
    /// </summary>
    [[nodiscard]] bool Contains(const Vec2i &aPoint) const;

    /// <summary>
    /// returns true if the specified point is within the roi area
    /// </summary>
    [[nodiscard]] bool Contains(int aX, int aY) const;

    /// <summary>
    /// returns true if the specified point is within the roi area
    /// </summary>
    [[nodiscard]] bool Contains(const Vec2f &aPoint) const;

    /// <summary>
    /// returns true if the specified point is within the roi area
    /// </summary>
    [[nodiscard]] bool Contains(const Vec2d &aPoint) const;

    /// <summary>
    /// returns true if the specified point is within the roi area
    /// </summary>
    [[nodiscard]] bool Contains(float aX, float aY) const;

    /// <summary>
    /// returns true if the specified point is within the roi area
    /// </summary>
    [[nodiscard]] bool Contains(double aX, double aY) const;

    /// <summary>
    /// returns true if aRoi is entirely within the roi area
    /// </summary>
    [[nodiscard]] bool Contains(const Roi &aRoi) const;

    /// <summary>
    /// Returns a roi that represents the intersection of this and the other roi.
    /// If there is no intersection, an empty roi is returned.
    /// </summary>
    [[nodiscard]] Roi Intersect(const Roi &aOther) const;

    /// <summary>
    /// Determines if this roi intersects with the other roi.
    /// </summary>
    [[nodiscard]] bool IntersectsWith(const Roi &aOther) const;

    /// <summary>
    /// Returns the center pixel of roi
    /// </summary>
    [[nodiscard]] Vec2i Center() const;

    /// <summary>
    /// Shifts the smaller roi until it entirely fits into the larger roi.<para/>
    /// If both have the same size, this is returned.
    /// </summary>
    [[nodiscard]] Roi ShiftUntilFit(const Roi &aOtherRoi) const;

    /// <summary>
    /// Enlarges the roi so that the new roi contains both source roi.
    /// </summary>
    [[nodiscard]] Roi Union(const Roi &aOtherRoi) const;
};

std::ostream &operator<<(std::ostream &aOs, const Roi &aRoi);
std::wostream &operator<<(std::wostream &aOs, const Roi &aRoi);
std::istream &operator>>(std::istream &aIs, Roi &aRoi);
std::wistream &operator>>(std::wistream &aIs, Roi &aRoi);

Roi operator+(const Roi &aLeft, const Vec2i &aRight);
Roi operator+(const Roi &aLeft, const Border &aRight);
Roi operator+(const Border &aLeft, const Roi &aRight);
// Roi operator+(const Roi &aLeft, const Roi &aRight);
Roi operator+(const Roi &aLeft, int aRight);
Roi operator+(int aLeft, const Roi &aRight);
Roi operator-(const Roi &aLeft, const Vec2i &aRight);
Roi operator-(const Roi &aLeft, const Border &aRight);
Border operator-(const Roi &aLeft, const Roi &aRight);
Roi operator-(const Roi &aLeft, int aRight);
// Roi operator-(int aLeft, const Roi &aRight);

Roi operator*(const Roi &aLeft, const Vec2i &aRight);
// Roi operator*(const Roi &aLeft, const Roi &aRight);
Roi operator*(const Roi &aLeft, int aRight);
Roi operator*(int aLeft, const Roi &aRight);
Roi operator/(const Roi &aLeft, const Vec2i &aRight);
// Roi operator/(const Roi &aLeft, const Roi &aRight);
Roi operator/(const Roi &aLeft, int aRight);
// Roi operator/(int aLeft, const Roi &aRight);

} // namespace opp::image