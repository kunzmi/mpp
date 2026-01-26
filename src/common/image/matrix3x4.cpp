#include "../dllexport_common.h"
#include "matrix3x4.h"
#include <algorithm>
#include <cassert>
#include <common/numberTypes.h>
#include <common/safeCast.h>
#include <concepts>
#include <cstddef>
#include <functional>

namespace mpp::image
{
template <RealFloatingPoint T> constexpr size_t Matrix3x4<T>::GetIndex(int aRow, int aCol)
{
    assert(aRow >= 0);
    assert(aCol >= 0);
    assert(aRow < mRows);
    assert(aCol < mCols);

    return to_size_t(aCol + aRow * mCols);
}

template <RealFloatingPoint T> Matrix3x4<T>::Matrix3x4(const T aValues[mSize]) noexcept : mData()
{
    std::copy(aValues, aValues + mSize, mData);
}

template <RealFloatingPoint T> Matrix3x4<T>::Matrix3x4(const T aValues[mRows][mCols]) noexcept : mData()
{
    std::copy(reinterpret_cast<const T *>(aValues), reinterpret_cast<const T *>(aValues) + mSize, mData);
}

// for conversion
template <RealFloatingPoint T>
template <RealFloatingPoint T2>
Matrix3x4<T>::Matrix3x4(const Matrix3x4<T2> &aOther) noexcept
    requires(!std::same_as<T, T2>)
{
    for (size_t i = 0; i < mSize; i++)
    {
        mData[i] = static_cast<T>(aOther.Data()[i]); // NOLINT
    }
}

template <RealFloatingPoint T> bool Matrix3x4<T>::operator==(const Matrix3x4 &aOther) const
{
    bool ret = true;
    for (size_t i = 0; i < mSize; i++)
    {
        ret &= mData[i] == aOther[i]; // NOLINT
    }
    return ret;
}

template <RealFloatingPoint T> bool Matrix3x4<T>::operator!=(const Matrix3x4 &aOther) const
{
    return !(*this == aOther);
}

template <RealFloatingPoint T> T &Matrix3x4<T>::operator()(int aRow, int aCol)
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT
}
template <RealFloatingPoint T> const T &Matrix3x4<T>::operator()(int aRow, int aCol) const
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT
}

template <RealFloatingPoint T> T &Matrix3x4<T>::operator[](int aFlatIndex)
{
    assert(aFlatIndex >= 0);
    assert(aFlatIndex < mSize);
    return mData[to_size_t(aFlatIndex)]; // NOLINT
}

template <RealFloatingPoint T> T &Matrix3x4<T>::operator[](size_t aFlatIndex)
{
    assert(to_int(aFlatIndex) < mSize);
    return mData[aFlatIndex]; // NOLINT
}

template <RealFloatingPoint T> const T *Matrix3x4<T>::Data() const
{
    return mData;
}

template <RealFloatingPoint T> T *Matrix3x4<T>::Data()
{
    return mData;
}

template <RealFloatingPoint T> Matrix3x4<T> &Matrix3x4<T>::operator+=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN += aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix3x4<T> &Matrix3x4<T>::operator+=(const Matrix3x4 &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::plus<>{});
    return *this;
}

template <RealFloatingPoint T> Matrix3x4<T> Matrix3x4<T>::operator+(const Matrix3x4 &aOther) const
{
    Matrix3x4 ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::plus<>{});
    return ret;
}

template <RealFloatingPoint T> Matrix3x4<T> &Matrix3x4<T>::operator-=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN -= aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix3x4<T> &Matrix3x4<T>::operator-=(const Matrix3x4 &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::minus<>{});
    return *this;
}

template <RealFloatingPoint T> Matrix3x4<T> Matrix3x4<T>::operator-(const Matrix3x4 &aOther) const
{
    Matrix3x4 ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::minus<>{});
    return ret;
}

template <RealFloatingPoint T> Matrix3x4<T> &Matrix3x4<T>::operator*=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN *= aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix3x4<T> &Matrix3x4<T>::operator/=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN /= aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix3x4<T> operator+(const Matrix3x4<T> &aLeft, T aRight)
{
    Matrix3x4 ret(aLeft);
    ret += aRight;
    return ret;
}
template <RealFloatingPoint T> Matrix3x4<T> operator+(T aLeft, const Matrix3x4<T> &aRight)
{
    Matrix3x4 ret(aLeft);
    ret += aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix3x4<T> operator-(const Matrix3x4<T> &aLeft, T aRight)
{
    Matrix3x4 ret(aLeft);
    ret -= aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix3x4<T> operator-(T aLeft, const Matrix3x4<T> &aRight)
{
    Matrix3x4 ret(aLeft);
    ret -= aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix3x4<T> operator*(const Matrix3x4<T> &aLeft, T aRight)
{
    Matrix3x4 ret(aLeft);
    ret *= aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix3x4<T> operator*(T aLeft, const Matrix3x4<T> &aRight)
{
    Matrix3x4 ret(aRight);
    ret *= aLeft;
    return ret;
}

template <RealFloatingPoint T> Matrix3x4<T> operator/(const Matrix3x4<T> &aLeft, T aRight)
{
    Matrix3x4 ret(aLeft);
    ret *= T(1) / aRight;
    return ret;
}

// instantiate for float and double:
template class Matrix3x4<float>;
template class Matrix3x4<double>;

template MPPEXPORT_COMMON Matrix3x4<float>::Matrix3x4(const Matrix3x4<double> &aOther);
template MPPEXPORT_COMMON Matrix3x4<double>::Matrix3x4(const Matrix3x4<float> &aOther);

template Matrix3x4<float> operator+(const Matrix3x4<float> &aLeft, float aRight);
template Matrix3x4<float> operator+(float aLeft, const Matrix3x4<float> &aRight);
template Matrix3x4<float> operator-(const Matrix3x4<float> &aLeft, float aRight);
template Matrix3x4<float> operator-(float aLeft, const Matrix3x4<float> &aRight);

template Matrix3x4<float> operator*(const Matrix3x4<float> &aLeft, float aRight);
template Matrix3x4<float> operator*(float aLeft, const Matrix3x4<float> &aRight);
template Matrix3x4<float> operator/(const Matrix3x4<float> &aLeft, float aRight);

template Matrix3x4<double> operator+(const Matrix3x4<double> &aLeft, double aRight);
template Matrix3x4<double> operator+(double aLeft, const Matrix3x4<double> &aRight);
template Matrix3x4<double> operator-(const Matrix3x4<double> &aLeft, double aRight);
template Matrix3x4<double> operator-(double aLeft, const Matrix3x4<double> &aRight);

template Matrix3x4<double> operator*(const Matrix3x4<double> &aLeft, double aRight);
template Matrix3x4<double> operator*(double aLeft, const Matrix3x4<double> &aRight);
template Matrix3x4<double> operator/(const Matrix3x4<double> &aLeft, double aRight);

} // namespace mpp::image