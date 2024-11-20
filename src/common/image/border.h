#pragma once
#include <common/defines.h>
#include <common/image/size2D.h>
#include <common/vector2.h>
#include <common/vector4.h>

namespace opp::image
{
/// <summary>
/// A Border defines the number of border pixels on each side of a 2D roi
/// Note: we avoid any absolute referencing like top/left or lower/right as an image can have any orientation. Instead
/// we only reference lower X -> higher X accoriding to the coordinate value.
/// Note: in device code, only the value members are available.
/// </summary>
struct alignas(4 * sizeof(int)) Border
{
    int lowerX{0};  // NOLINT(misc-non-private-member-variables-in-classes)
    int lowerY{0};  // NOLINT(misc-non-private-member-variables-in-classes)
    int higherX{0}; // NOLINT(misc-non-private-member-variables-in-classes)
    int higherY{0}; // NOLINT(misc-non-private-member-variables-in-classes)

    /// <summary>
    /// Initializes a new border to all components = 0
    /// </summary>
    Border() noexcept = default;

    /// <summary>
    /// Initializes a new border to (aLowerX, aLowerY, aHigherX, aHigherY)
    /// </summary>
    Border(int aLowerX, int aLowerY, int aHigherX, int aHigherY) noexcept;

    /// <summary>
    /// Initializes a new border to (aX, aY, aX,aY)
    /// </summary>
    Border(int aX, int aY) noexcept;

    /// <summary>
    /// Initializes a new border to (aSize, aSize, aSize, aSize)
    /// </summary>
    Border(int aSize) noexcept; // NOLINT(hicpp-explicit-conversions)

    /// <summary>
    /// Initializes a new border to (aVec.x, aVec.y, aVec.z, aVec.w)
    /// </summary>
    Border(const Vector4<int> &aVec) noexcept; // NOLINT(hicpp-explicit-conversions)

    /// <summary>
    /// Initializes a new border to (aArr[0], aArr[1], aArr[2], aArr[3])
    /// </summary>
    Border(int aArr[4]) noexcept; // NOLINT(hicpp-explicit-conversions)

    ~Border() = default;

    Border(const Border &)                     = default;
    Border(Border &&)                          = default;
    Border &operator=(const Border &) noexcept = default;
    Border &operator=(Border &&) noexcept      = default;

    /// <summary>
    /// Enlarges the border by the specified amount on all sides
    /// </summary>
    Border &operator+=(int aOther);

    /// <summary>
    /// Enlarges the border by the given border.
    /// </summary>
    Border &operator+=(const Border &aOther);

    /// <summary>
    /// Reduces the border by the specified amount on all sides
    /// </summary>
    Border &operator-=(int aOther);

    /// <summary>
    /// Enlarges the border by the specified amount, different values for X and Y
    /// </summary>
    Border &operator+=(const Vector2<int> &aOther);

    /// <summary>
    /// Reduces the border by the specified amount, different values for X and Y
    /// </summary>
    Border &operator-=(const Vector2<int> &aOther);

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    Border &operator*=(int aOther);

    /// <summary>
    /// Component wise division
    /// </summary>
    Border &operator/=(int aOther);

    /// <summary>
    /// returns true if all border components are equal
    /// </summary>
    bool operator==(const Border &aOther) const;

    /// <summary>
    /// returns true if at least one border component is not equal
    /// </summary>
    bool operator!=(const Border &aOther) const;
};

Border operator+(const Border &aLeft, const Vector2<int> &aRight);
Border operator+(const Border &aLeft, const Border &aRight);
Border operator+(const Border &aLeft, int aRight);
Border operator+(int aLeft, const Border &aRight);
Border operator-(const Border &aLeft, const Vector2<int> &aRight);
Border operator-(const Border &aLeft, const Border &aRight);
Border operator-(const Border &aLeft, int aRight);
Border operator-(int aLeft, const Border &aRight);

Border operator*(const Border &aLeft, const Vector2<int> &aRight);
Border operator*(const Border &aLeft, const Border &aRight);
Border operator*(const Border &aLeft, int aRight);
Border operator*(int aLeft, const Border &aRight);
Border operator/(const Border &aLeft, const Vector2<int> &aRight);
Border operator/(const Border &aLeft, const Border &aRight);
Border operator/(const Border &aLeft, int aRight);
Border operator/(int aLeft, const Border &aRight);

std::ostream &operator<<(std::ostream &aOs, const Border &aBorder);
std::wostream &operator<<(std::wostream &aOs, const Border &aBorder);
std::istream &operator>>(std::istream &aIs, Border &aBorder);
std::wistream &operator>>(std::wistream &aIs, Border &aBorder);

} // namespace opp::image