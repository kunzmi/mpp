#include "border.h"
#include "roi.h"
#include "size2D.h"
#include <common/safeCast.h>
#include <common/vector2.h>
#include <common/vector4.h>
#include <istream>
#include <ostream>

namespace opp::image
{
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Roi::Roi(int aX, int aY, int aWidth, int aHeight) noexcept : x(aX), y(aY), width(aWidth), height(aHeight)
{
}
Roi::Roi(int aX, int aY, const Size2D &aSize) noexcept : x(aX), y(aY), width(aSize.x), height(aSize.y)
{
}
Roi::Roi(const Vector2<int> &aPos, const Size2D &aSize) noexcept : x(aPos.x), y(aPos.y), width(aSize.x), height(aSize.y)
{
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Roi::Roi(const Vector2<int> &aPos, int aWidth, int aHeight) noexcept
    : x(aPos.x), y(aPos.y), width(aWidth), height(aHeight)
{
}
Roi::Roi(const Vector4<int> &aVec) noexcept : x(aVec.x), y(aVec.y), width(aVec.z), height(aVec.w)
{
}
Roi::Roi(int aArr[4]) noexcept : x(aArr[0]), y(aArr[1]), width(aArr[2]), height(aArr[3])
{
}
int Roi::FirstX() const
{
    return x;
}
int Roi::FirstY() const
{
    return y;
}
int Roi::LastX() const
{
    return x + width - 1;
}
int Roi::LastY() const
{
    return y + height - 1;
}
Vector2<int> Roi::FirstPixel() const
{
    return Vector2<int>{FirstX(), FirstY()};
}
Vector2<int> Roi::LastPixel() const
{
    return Vector2<int>{LastX(), LastY()};
}
Vector2<int> Roi::BoundingBoxMin() const
{
    return FirstPixel();
}
Vector2<int> Roi::BoundingBoxMax() const
{
    return Vector2<int>{x + width, y + height};
}
Roi &Roi::operator+=(int aOther)
{
    x -= aOther;
    y -= aOther;
    width += 2 * aOther;
    height += 2 * aOther;
    return *this;
}
Roi &Roi::operator+=(const Vector2<int> &aOther)
{
    x -= aOther.x;
    y -= aOther.y;
    width += 2 * aOther.x;
    height += 2 * aOther.y;
    return *this;
}
Roi &Roi::operator+=(const Border &aOther)
{
    x -= aOther.lowerX;
    y -= aOther.lowerY;
    width += aOther.lowerX + aOther.higherX;
    height += aOther.lowerY + aOther.higherY;
    return *this;
}
Roi &Roi::operator+=(const Roi &aOther)
{
    *this = Union(aOther);
    return *this;
}
Roi &Roi::operator-=(int aOther)
{
    x += aOther;
    y += aOther;
    width -= 2 * aOther;
    height -= 2 * aOther;
    return *this;
}
Roi &Roi::operator-=(const Border &aOther)
{
    x += aOther.lowerX;
    y += aOther.lowerY;
    width -= aOther.lowerX + aOther.higherX;
    height -= aOther.lowerY + aOther.higherY;
    return *this;
}
Roi &Roi::operator-=(const Vector2<int> &aOther)
{
    x += aOther.x;
    y += aOther.y;
    width -= 2 * aOther.x;
    height -= 2 * aOther.y;
    return *this;
}
Roi &Roi::operator*=(int aOther)
{
    x *= aOther;
    y *= aOther;
    width *= aOther;
    height *= aOther;
    return *this;
}
Roi &Roi::operator/=(int aOther)
{
    x /= aOther;
    y /= aOther;
    width /= aOther;
    height /= aOther;
    return *this;
}
bool Roi::operator==(const Roi &aOther) const
{
    return x == aOther.x && y == aOther.y && width == aOther.width && height == aOther.height;
}
bool Roi::operator!=(const Roi &aOther) const
{
    return x != aOther.x || y != aOther.y || width != aOther.width || height != aOther.height;
}
bool Roi::Contains(const Vector2<int> &aPoint) const
{
    return Contains(aPoint.x, aPoint.y);
}
bool Roi::Contains(int aX, int aY) const
{
    return (aX >= FirstX()) && (aX <= LastX()) && (aY >= FirstY()) && (aY <= LastY());
}
bool Roi::Contains(const Vector2<float> &aPoint) const
{
    return Contains(aPoint.x, aPoint.y);
}
bool Roi::Contains(float aX, float aY) const
{
    return (aX >= to_float(FirstX())) && (aX <= to_float(LastX())) && (aY >= to_float(FirstY())) &&
           (aY <= to_float(LastY()));
}
bool Roi::Contains(const Roi &aRoi) const
{
    return Contains(aRoi.FirstPixel()) && Contains(aRoi.LastPixel());
}
Roi Roi::Intersect(const Roi &aOther) const
{
    int iX = FirstX();
    if (iX < aOther.FirstX())
    {
        iX = aOther.FirstX();
    }

    int iY = FirstY();
    if (iY < aOther.FirstY())
    {
        iY = aOther.FirstY();
    }

    int iX2 = LastX();
    if (iX2 > aOther.LastX())
    {
        iX2 = aOther.LastX();
    }

    int iY2 = LastY();
    if (iY2 > aOther.LastY())
    {
        iY2 = aOther.LastY();
    }

    int iWidth  = iX2 - iX + 1;
    int iHeight = iY2 - iY + 1;
    if (iWidth <= 0 || iHeight <= 0)
    {
        iX      = 0;
        iY      = 0;
        iWidth  = 0;
        iHeight = 0;
    }
    return Roi{iX, iY, iWidth, iHeight};
}
bool Roi::IntersectsWith(const Roi &aOther) const
{
    const Vector2<int> firstPixel = aOther.FirstPixel();
    const Vector2<int> lastPixel  = aOther.LastPixel();
    const Vector2<int> firstXlastY{firstPixel.x, lastPixel.y};
    const Vector2<int> lastXfirstY{lastPixel.x, firstPixel.y};
    return (Contains(firstPixel) || Contains(lastPixel) || aOther.Contains(firstXlastY) ||
            aOther.Contains(lastXfirstY));
}
Vector2<int> Roi::Center() const
{
    return {x + width / 2, y + height / 2};
}
Roi Roi::ShiftUntilFit(const Roi &aOtherRoi) const
{
    Roi small;
    Roi big;
    if (this->width < aOtherRoi.width && this->height < aOtherRoi.height)
    {
        small = *this;
        big   = aOtherRoi;
    }
    else
    {
        if (this->width > aOtherRoi.width && this->height > aOtherRoi.height)
        {
            small = aOtherRoi;
            big   = *this;
        }
        else
        {
            return *this;
        }
    }
    // if it doesn't fit
    if (!big.Contains(small))
    {
        // x dimension:
        int diffX = small.FirstX() - big.FirstX();
        if (diffX < 0)
        {
            small.x -= diffX;
        }
        diffX = small.LastX() - big.LastX();
        if (diffX > 0)
        {
            small.x -= diffX;
        }
        // y dimension:
        int diffY = small.FirstY() - big.FirstY();
        if (diffY < 0)
        {
            small.y -= diffY;
        }
        diffY = small.LastY() - big.LastY();
        if (diffY > 0)
        {
            small.y -= diffY;
        }
    }
    return small;
}
Roi Roi::Union(const Roi &aOtherRoi) const
{
    int iX = FirstX();
    if (iX > aOtherRoi.FirstX())
    {
        iX = aOtherRoi.FirstX();
    }

    int iY = FirstX();
    if (iY > aOtherRoi.FirstX())
    {
        iY = aOtherRoi.FirstX();
    }

    int iX2 = LastX();
    if (iX2 < aOtherRoi.LastX())
    {
        iX2 = aOtherRoi.LastX();
    }

    int iY2 = LastY();
    if (iY2 < aOtherRoi.LastY())
    {
        iY2 = aOtherRoi.LastY();
    }

    int iWidth  = iX2 - iX + 1;
    int iHeight = iY2 - iY + 1;
    if (iWidth <= 0 || iHeight <= 0)
    {
        iX      = 0;
        iY      = 0;
        iWidth  = 0;
        iHeight = 0;
    }
    return Roi{iX, iY, iWidth, iHeight};
}
std::ostream &operator<<(std::ostream &aOs, const Roi &aRoi)
{
    aOs << "ROI: X = [" << aRoi.FirstX() << ":" << aRoi.LastX() << "] Y = [" << aRoi.FirstY() << ":" << aRoi.LastY()
        << "]";
    return aOs;
}
std::wostream &operator<<(std::wostream &aOs, const Roi &aRoi)
{
    aOs << "ROI: X = [" << aRoi.FirstX() << ":" << aRoi.LastX() << "] Y = [" << aRoi.FirstY() << ":" << aRoi.LastY()
        << "]";
    return aOs;
}
std::istream &operator>>(std::istream &aIs, Roi &aRoi)
{
    aIs >> aRoi.x >> aRoi.y >> aRoi.width >> aRoi.height;
    return aIs;
}
std::wistream &operator>>(std::wistream &aIs, Roi &aRoi)
{
    aIs >> aRoi.x >> aRoi.y >> aRoi.width >> aRoi.height;
    return aIs;
}
Roi operator+(const Roi &aLeft, const Vector2<int> &aRight)
{
    return Roi{aLeft.x - aRight.x, aLeft.y - aRight.y, aLeft.width + 2 * aRight.x, aLeft.height + 2 * aRight.y};
}
Roi operator+(const Roi &aLeft, const Border &aRight)
{
    return Roi{aLeft.x - aRight.lowerX, aLeft.y - aRight.lowerY, aLeft.width + aRight.lowerX + aRight.higherX,
               aLeft.height + aRight.lowerY + aRight.higherY};
}
Roi operator+(const Border &aLeft, const Roi &aRight)
{
    return Roi{aRight.x - aLeft.lowerX, aRight.y - aLeft.lowerY, aRight.width + aLeft.lowerX + aLeft.higherX,
               aRight.height + aLeft.lowerY + aLeft.higherY};
}
Roi operator+(const Roi &aLeft, int aRight)
{
    return Roi{aLeft.x - aRight, aLeft.y - aRight, aLeft.width + 2 * aRight, aLeft.height + 2 * aRight};
}
Roi operator+(int aLeft, const Roi &aRight)
{
    return Roi{aLeft - aRight.x, aLeft - aRight.y, 2 * aLeft + aRight.width, 2 * aLeft + aRight.height};
}
Roi operator-(const Roi &aLeft, const Vector2<int> &aRight)
{
    return Roi{aLeft.x + aRight.x, aLeft.y + aRight.y, aLeft.width - 2 * aRight.x, aLeft.height - 2 * aRight.y};
}
Roi operator-(const Roi &aLeft, const Border &aRight)
{
    return Roi{aLeft.x + aRight.lowerX, aLeft.y + aRight.lowerY, aLeft.width - aRight.lowerX - aRight.higherX,
               aLeft.height - aRight.lowerY - aRight.higherY};
}
Border operator-(const Roi &aLeft, const Roi &aRight)
{
    return {aLeft.FirstX() - aRight.FirstX(), aLeft.FirstY() - aRight.FirstY(), aLeft.LastX() - aRight.LastX(),
            aLeft.LastY() - aRight.LastY()};
}
Roi operator-(const Roi &aLeft, int aRight)
{
    return Roi{aLeft.x + aRight, aLeft.y + aRight, aLeft.width - 2 * aRight, aLeft.height - 2 * aRight};
}
Roi operator-(int aLeft, const Roi &aRight)
{
    return Roi{aLeft + aRight.x, aLeft + aRight.y, 2 * aLeft - aRight.width, 2 * aLeft - aRight.height};
}
Roi operator*(const Roi &aLeft, const Vector2<int> &aRight)
{
    return Roi{aLeft.x * aRight.x, aLeft.y * aRight.y, aLeft.width * aRight.x, aLeft.height * aRight.y};
}
Roi operator*(const Roi &aLeft, int aRight)
{
    return Roi{aLeft.x * aRight, aLeft.y * aRight, aLeft.width * aRight, aLeft.height * aRight};
}
Roi operator*(int aLeft, const Roi &aRight)
{
    return Roi{aLeft * aRight.x, aLeft * aRight.y, aLeft * aRight.width, aLeft * aRight.height};
}
Roi operator/(const Roi &aLeft, const Vector2<int> &aRight)
{
    return Roi{aLeft.x / aRight.x, aLeft.y / aRight.y, aLeft.width / aRight.x, aLeft.height / aRight.y};
}
Roi operator/(const Roi &aLeft, int aRight)
{
    return Roi{aLeft.x / aRight, aLeft.y / aRight, aLeft.width / aRight, aLeft.height / aRight};
}
Roi operator/(int aLeft, const Roi &aRight)
{
    return Roi{aLeft / aRight.x, aLeft / aRight.y, aLeft / aRight.width, aLeft / aRight.height};
}
} // namespace opp::image