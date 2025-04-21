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

Size2D::iterator Size2D::begin() const
{
    return {Index(), x};
}

Size2D::iterator Size2D::end() const
{
    Index idx;
    idx.Flat  = to_size_t(x) * to_size_t(y);
    idx.Pixel = Vector2<int>(0, y);
    return {idx, x};
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

Size2D::iterator::iterator(const Size2D::Index &aValue, int aWidth) : mValue(aValue), mWidth(aWidth)
{
}

Size2D::iterator &Size2D::iterator::operator++()
{
    mValue.Flat++;
    mValue.Pixel.x++;
    if (mValue.Pixel.x >= mWidth)
    {
        mValue.Pixel.x = 0;
        mValue.Pixel.y++;
    }
    return *this;
}
Size2D::iterator &Size2D::iterator::operator--()
{
    mValue.Flat--;
    mValue.Pixel.x--;
    if (mValue.Pixel.x < 0)
    {
        mValue.Pixel.x = mWidth - 1;
        mValue.Pixel.y--;
    }
    return *this;
}

Size2D::iterator Size2D::iterator::operator++(int) & // NOLINT(cert-dcl21-cpp)
{
    iterator ret = *this;
    operator++();
    return ret;
}
Size2D::iterator Size2D::iterator::operator--(int) & // NOLINT(cert-dcl21-cpp)
{
    iterator ret = *this;
    operator--();
    return ret;
}

bool Size2D::iterator::operator==(Size2D::iterator const &aOther) const
{
    return mValue.Flat == aOther.mValue.Flat && mValue.Pixel == aOther.mValue.Pixel && mWidth == aOther.mWidth;
}

bool Size2D::iterator::operator!=(Size2D::iterator const &aOther) const
{
    return !(*this == aOther);
}

Size2D::iterator::reference Size2D::iterator::operator*()
{
    return mValue;
}

Size2D::iterator::pointer Size2D::iterator::operator->()
{
    return &mValue;
}

Size2D::iterator::difference_type Size2D::iterator::operator[](Size2D::iterator::difference_type aRhs) const
{
    difference_type diff = to_long64(mValue.Flat);
    diff += aRhs;
    return diff;
}

Size2D::iterator::difference_type Size2D::iterator::operator-(const Size2D::iterator &aRhs) const
{
    difference_type diff = to_long64(mValue.Flat);
    diff -= to_long64(aRhs.mValue.Flat);
    return diff;
}
Size2D::iterator Size2D::iterator::operator+(Size2D::iterator::difference_type aRhs) const
{
    Size2D::Index idx;
    idx.Flat    = to_size_t(to_long64(mValue.Flat) + aRhs);
    idx.Pixel.y = to_int(idx.Flat / to_size_t(mWidth));
    idx.Pixel.x = to_int(idx.Flat - to_size_t(idx.Pixel.y) * to_size_t(mWidth));
    return {idx, mWidth};
}
Size2D::iterator Size2D::iterator::operator-(Size2D::iterator::difference_type aRhs) const
{
    Size2D::Index idx;
    idx.Flat    = to_size_t(to_long64(mValue.Flat) - aRhs);
    idx.Pixel.y = to_int(idx.Flat / to_size_t(mWidth));
    idx.Pixel.x = to_int(idx.Flat - to_size_t(idx.Pixel.y) * to_size_t(mWidth));
    return {idx, mWidth};
}
Size2D::iterator &Size2D::iterator::operator+=(Size2D::iterator::difference_type aRhs)
{
    mValue.Flat    = to_size_t(to_long64(mValue.Flat) + aRhs);
    mValue.Pixel.y = to_int(mValue.Flat / to_size_t(mWidth));
    mValue.Pixel.x = to_int(mValue.Flat - to_size_t(mValue.Pixel.y) * to_size_t(mWidth));
    return *this;
}
Size2D::iterator &Size2D::iterator::operator-=(Size2D::iterator::difference_type aRhs)
{
    mValue.Flat    = to_size_t(to_long64(mValue.Flat) - aRhs);
    mValue.Pixel.y = to_int(mValue.Flat / to_size_t(mWidth));
    mValue.Pixel.x = to_int(mValue.Flat - to_size_t(mValue.Pixel.y) * to_size_t(mWidth));
    return *this;
}
Size2D::iterator operator+(Size2D::iterator::difference_type aLhs, const Size2D::iterator &aRhs)
{
    Size2D::Index idx;
    idx.Flat    = to_size_t(aLhs + to_long64(aRhs.mValue.Flat));
    idx.Pixel.y = to_int(idx.Flat / to_size_t(aRhs.mWidth));
    idx.Pixel.x = to_int(idx.Flat - to_size_t(idx.Pixel.y) * to_size_t(aRhs.mWidth));
    return {idx, aRhs.mWidth};
}
Size2D::iterator operator-(Size2D::iterator::difference_type aLhs, const Size2D::iterator &aRhs)
{
    Size2D::Index idx;
    idx.Flat    = to_size_t(aLhs - to_long64(aRhs.mValue.Flat));
    idx.Pixel.y = to_int(idx.Flat / to_size_t(aRhs.mWidth));
    idx.Pixel.x = to_int(idx.Flat - to_size_t(idx.Pixel.y) * to_size_t(aRhs.mWidth));
    return {idx, aRhs.mWidth};
}

bool Size2D::iterator::operator>(const Size2D::iterator &aRhs) const
{
    return mValue.Flat > aRhs.mValue.Flat;
}
bool Size2D::iterator::operator<(const Size2D::iterator &aRhs) const
{
    return mValue.Flat < aRhs.mValue.Flat;
}
bool Size2D::iterator::operator>=(const Size2D::iterator &aRhs) const
{
    return mValue.Flat >= aRhs.mValue.Flat;
}
bool Size2D::iterator::operator<=(const Size2D::iterator &aRhs) const
{
    return mValue.Flat <= aRhs.mValue.Flat;
}
} // namespace opp::image