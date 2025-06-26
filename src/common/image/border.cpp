#include "border.h"
#include <common/vectorTypes.h>
#include <istream>
#include <ostream>

namespace mpp::image
{
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Border::Border(int aLowerX, int aLowerY, int aHigherX, int aHigherY) noexcept
    : lowerX(aLowerX), lowerY(aLowerY), higherX(aHigherX), higherY(aHigherY)
{
}
Border::Border(int aX, int aY) noexcept : lowerX(aX), lowerY(aY), higherX(aX), higherY(aY)
{
}
Border::Border(int aSize) noexcept : lowerX(aSize), lowerY(aSize), higherX(aSize), higherY(aSize)
{
}
Border::Border(const Vec4i &aVec) noexcept : lowerX(aVec.x), lowerY(aVec.y), higherX(aVec.z), higherY(aVec.w)
{
}
Border::Border(int aArr[4]) noexcept : lowerX(aArr[0]), lowerY(aArr[1]), higherX(aArr[2]), higherY(aArr[3])
{
}
Border &Border::operator+=(int aOther)
{
    lowerX += aOther;
    lowerY += aOther;
    higherX += aOther;
    higherY += aOther;
    return *this;
}
Border &Border::operator+=(const Border &aOther)
{
    lowerX += aOther.lowerX;
    lowerY += aOther.lowerY;
    higherX += aOther.higherX;
    higherY += aOther.higherY;
    return *this;
}
Border &Border::operator-=(int aOther)
{
    lowerX -= aOther;
    lowerY -= aOther;
    higherX -= aOther;
    higherY -= aOther;
    return *this;
}
Border &Border::operator+=(const Vec2i &aOther)
{
    lowerX += aOther.x;
    lowerY += aOther.y;
    higherX += aOther.x;
    higherY += aOther.y;
    return *this;
}
Border &Border::operator-=(const Vec2i &aOther)
{
    lowerX -= aOther.x;
    lowerY -= aOther.y;
    higherX -= aOther.x;
    higherY -= aOther.y;
    return *this;
}
Border &Border::operator*=(int aOther)
{
    lowerX *= aOther;
    lowerY *= aOther;
    higherX *= aOther;
    higherY *= aOther;
    return *this;
}
Border &Border::operator/=(int aOther)
{
    lowerX /= aOther;
    lowerY /= aOther;
    higherX /= aOther;
    higherY /= aOther;
    return *this;
}
bool Border::operator==(const Border &aOther) const
{
    return lowerX == aOther.lowerX && lowerY == aOther.lowerY && higherX == aOther.higherX && higherY == aOther.higherY;
}
bool Border::operator!=(const Border &aOther) const
{
    return lowerX != aOther.lowerX || lowerY != aOther.lowerY || higherX != aOther.higherX || higherY != aOther.higherY;
}
std::ostream &operator<<(std::ostream &aOs, const Border &aBorder)
{
    aOs << '(' << aBorder.lowerX << ", " << aBorder.lowerY << ", " << aBorder.higherX << ", " << aBorder.higherY << ')';
    return aOs;
}
std::wostream &operator<<(std::wostream &aOs, const Border &aBorder)
{
    aOs << '(' << aBorder.lowerX << ", " << aBorder.lowerY << ", " << aBorder.higherX << ", " << aBorder.higherY << ')';
    return aOs;
}
std::istream &operator>>(std::istream &aIs, Border &aBorder)
{
    aIs >> aBorder.lowerX >> aBorder.lowerY >> aBorder.higherX >> aBorder.higherY;
    return aIs;
}
std::wistream &operator>>(std::wistream &aIs, Border &aBorder)
{
    aIs >> aBorder.lowerX >> aBorder.lowerY >> aBorder.higherX >> aBorder.higherY;
    return aIs;
}
Border operator+(const Border &aLeft, const Vec2i &aRight)
{

    return {aLeft.lowerX + aRight.x, aLeft.lowerY + aRight.y, aLeft.higherX + aRight.x, aLeft.higherY + aRight.y};
}
Border operator+(const Border &aLeft, const Border &aRight)
{
    return {aLeft.lowerX + aRight.lowerX, aLeft.lowerY + aRight.lowerY, aLeft.higherX + aRight.higherX,
            aLeft.higherY + aRight.higherY};
}
Border operator+(const Border &aLeft, int aRight)
{
    return {aLeft.lowerX + aRight, aLeft.lowerY + aRight, aLeft.higherX + aRight, aLeft.higherY + aRight};
}
Border operator+(int aLeft, const Border &aRight)
{
    return {aLeft + aRight.lowerX, aLeft + aRight.lowerY, aLeft + aRight.higherX, aLeft + aRight.higherY};
}
Border operator-(const Border &aLeft, const Vec2i &aRight)
{
    return {aLeft.lowerX - aRight.x, aLeft.lowerY - aRight.y, aLeft.higherX - aRight.x, aLeft.higherY - aRight.y};
}
Border operator-(const Border &aLeft, const Border &aRight)
{
    return {aLeft.lowerX - aRight.lowerX, aLeft.lowerY - aRight.lowerY, aLeft.higherX - aRight.higherX,
            aLeft.higherY - aRight.higherY};
}
Border operator-(const Border &aLeft, int aRight)
{
    return {aLeft.lowerX - aRight, aLeft.lowerY - aRight, aLeft.higherX - aRight, aLeft.higherY - aRight};
}
Border operator-(int aLeft, const Border &aRight)
{
    return {aLeft - aRight.lowerX, aLeft - aRight.lowerY, aLeft - aRight.higherX, aLeft - aRight.higherY};
}
Border operator*(const Border &aLeft, const Vec2i &aRight)
{
    return {aLeft.lowerX * aRight.x, aLeft.lowerY * aRight.y, aLeft.higherX * aRight.x, aLeft.higherY * aRight.y};
}
Border operator*(const Border &aLeft, const Border &aRight)
{
    return {aLeft.lowerX * aRight.lowerX, aLeft.lowerY * aRight.lowerY, aLeft.higherX * aRight.higherX,
            aLeft.higherY * aRight.higherY};
}
Border operator*(const Border &aLeft, int aRight)
{
    return {aLeft.lowerX * aRight, aLeft.lowerY * aRight, aLeft.higherX * aRight, aLeft.higherY * aRight};
}
Border operator*(int aLeft, const Border &aRight)
{
    return {aLeft * aRight.lowerX, aLeft * aRight.lowerY, aLeft * aRight.higherX, aLeft * aRight.higherY};
}
Border operator/(const Border &aLeft, const Vec2i &aRight)
{
    return {aLeft.lowerX / aRight.x, aLeft.lowerY / aRight.y, aLeft.higherX / aRight.x, aLeft.higherY / aRight.y};
}
Border operator/(const Border &aLeft, const Border &aRight)
{
    return {aLeft.lowerX / aRight.lowerX, aLeft.lowerY / aRight.lowerY, aLeft.higherX / aRight.higherX,
            aLeft.higherY / aRight.higherY};
}
Border operator/(const Border &aLeft, int aRight)
{
    return {aLeft.lowerX / aRight, aLeft.lowerY / aRight, aLeft.higherX / aRight, aLeft.higherY / aRight};
}
Border operator/(int aLeft, const Border &aRight)
{
    return {aLeft / aRight.lowerX, aLeft / aRight.lowerY, aLeft / aRight.higherX, aLeft / aRight.higherY};
}
} // namespace mpp::image