#include "size2D.h"
#include <common/safeCast.h>
#include <common/vector2.h>
#include <cstddef>
#include <istream>
#include <ostream>

namespace opp::image
{
Size2D::Size2D() noexcept : Vector2<int>(0)
{
}

Size2D::Size2D(int aX, int aY) noexcept : Vector2<int>(aX, aY)
{
}

Size2D::Size2D(const Vector2<int> &aVec) noexcept : Vector2<int>(aVec)
{
}

Size2D::Size2D(int aArr[2]) noexcept : Vector2<int>(aArr)
{
}

Size2D &Size2D::operator=(const Vector2<int> &aOther) noexcept
{
    x = aOther.x;
    y = aOther.y;
    return *this;
}

Size2D &Size2D::operator+=(int aOther)
{
    x += aOther;
    y += aOther;
    return *this;
}

Size2D &Size2D::operator-=(int aOther)
{
    x -= aOther;
    y -= aOther;
    return *this;
}

Size2D &Size2D::operator*=(int aOther)
{
    x *= aOther;
    y *= aOther;
    return *this;
}

Size2D &Size2D::operator/=(int aOther)
{
    x /= aOther;
    y /= aOther;
    return *this;
}

Size2D &Size2D::operator+=(const Vector2<int> &aOther)
{
    x += aOther.x;
    y += aOther.y;
    return *this;
}

Size2D &Size2D::operator-=(const Vector2<int> &aOther)
{
    x -= aOther.x;
    y -= aOther.y;
    return *this;
}

Size2D &Size2D::operator*=(const Vector2<int> &aOther)
{
    x *= aOther.x;
    y *= aOther.y;
    return *this;
}

Size2D &Size2D::operator/=(const Vector2<int> &aOther)
{
    x /= aOther.x;
    y /= aOther.y;
    return *this;
}

Size2D operator+(const Size2D &aLeft, const Vector2<int> &aRight)
{
    return Size2D{aLeft.x + aRight.x, aLeft.y + aRight.y};
}

Size2D operator+(const Size2D &aLeft, const Size2D &aRight)
{
    return Size2D{aLeft.x + aRight.x, aLeft.y + aRight.y};
}

Size2D operator+(const Size2D &aLeft, int aRight)
{
    return Size2D{aLeft.x + aRight, aLeft.y + aRight};
}

Size2D operator+(int aLeft, const Size2D &aRight)
{
    return Size2D{aLeft + aRight.x, aLeft + aRight.y};
}

Size2D operator-(const Size2D &aLeft, const Vector2<int> &aRight)
{
    return Size2D{aLeft.x - aRight.x, aLeft.y - aRight.y};
}

Size2D operator-(const Size2D &aLeft, const Size2D &aRight)
{
    return Size2D{aLeft.x - aRight.x, aLeft.y - aRight.y};
}

Size2D operator-(const Size2D &aLeft, int aRight)
{
    return Size2D{aLeft.x - aRight, aLeft.y - aRight};
}

Size2D operator-(int aLeft, const Size2D &aRight)
{
    return Size2D{aLeft - aRight.x, aLeft - aRight.y};
}

Size2D operator*(const Size2D &aLeft, const Vector2<int> &aRight)
{
    return Size2D{aLeft.x * aRight.x, aLeft.y * aRight.y};
}

Size2D operator*(const Size2D &aLeft, const Size2D &aRight)
{
    return Size2D{aLeft.x * aRight.x, aLeft.y * aRight.y};
}

Size2D operator*(const Size2D &aLeft, int aRight)
{
    return Size2D{aLeft.x * aRight, aLeft.y * aRight};
}

Size2D operator*(int aLeft, const Size2D &aRight)
{
    return Size2D{aLeft * aRight.x, aLeft * aRight.y};
}

Size2D operator/(const Size2D &aLeft, const Vector2<int> &aRight)
{
    return Size2D{aLeft.x / aRight.x, aLeft.y / aRight.y};
}

Size2D operator/(const Size2D &aLeft, const Size2D &aRight)
{
    return Size2D{aLeft.x / aRight.x, aLeft.y / aRight.y};
}

Size2D operator/(const Size2D &aLeft, int aRight)
{
    return Size2D{aLeft.x / aRight, aLeft.y / aRight};
}

Size2D operator/(int aLeft, const Size2D &aRight)
{
    return Size2D{aLeft / aRight.x, aLeft / aRight.y};
}

size_t Size2D::operator()(const Vector2<int> &aCoords) const
{
    return GetFlatIndex(aCoords);
}

size_t Size2D::operator()(int aX, int aY) const
{
    return GetFlatIndex(aX, aY);
}

size_t Size2D::operator()(size_t aX, size_t aY) const
{
    return GetFlatIndex(aX, aY);
}

Vector2<int> Size2D::GetCoordinates(size_t aIndex) const
{
    Vector2<int> ret; // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    ret.y = to_int(aIndex / to_size_t(x));
    ret.x = to_int(aIndex - to_size_t(ret.y) * to_size_t(x));

    return ret;
}

Vector2<int> Size2D::GetCoordinates(int aIndex) const
{
    return GetCoordinates(to_size_t(aIndex));
}

size_t Size2D::GetFlatIndex(const Vector2<int> &aCoords) const
{
    return to_size_t(aCoords.y) * to_size_t(x) + to_size_t(aCoords.x);
}

size_t Size2D::GetFlatIndex(int aX, int aY) const
{
    return to_size_t(aY) * to_size_t(x) + to_size_t(aX);
}

size_t Size2D::GetFlatIndex(size_t aX, size_t aY) const
{
    return aY * to_size_t(x) + aX;
}

size_t Size2D::TotalSize() const
{
    return to_size_t(x) * to_size_t(y);
}

std::ostream &operator<<(std::ostream &aOs, const Size2D &aSize)
{
    aOs << '(' << aSize.x << " x " << aSize.y << ')';
    return aOs;
}

std::wostream &operator<<(std::wostream &aOs, const Size2D &aSize)
{
    aOs << '(' << aSize.x << " x " << aSize.y << ')';
    return aOs;
}

std::istream &operator>>(std::istream &aIs, Size2D &aSize)
{
    aIs >> aSize.x >> aSize.y;
    return aIs;
}

std::wistream &operator>>(std::wistream &aIs, Size2D &aSize)
{
    aIs >> aSize.x >> aSize.y;
    return aIs;
}
} // namespace opp::image