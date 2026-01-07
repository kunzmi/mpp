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

// template <RealFloatingPoint T> Matrix3x4<T>::Matrix3x4() noexcept : mData()
//{
//     mData[0]  = T(1);
//     mData[5]  = T(1);
//     mData[10] = T(1);
//     mData[1]  = T(0);
//     mData[2]  = T(0);
//     mData[3]  = T(0);
//     mData[4]  = T(0);
//     mData[6]  = T(0);
//     mData[7]  = T(0);
//     mData[8]  = T(0);
//     mData[9]  = T(0);
//     mData[11] = T(0);
// }

// template <RealFloatingPoint T> Matrix3x4<T>::Matrix3x4(T aX) noexcept : mData()
//{
//     mData[0]  = aX;
//     mData[1]  = aX;
//     mData[2]  = aX;
//     mData[3]  = aX;
//     mData[4]  = aX;
//     mData[5]  = aX;
//     mData[6]  = aX;
//     mData[7]  = aX;
//     mData[8]  = aX;
//     mData[9]  = aX;
//     mData[10] = aX;
//     mData[11] = aX;
// }

template <RealFloatingPoint T> Matrix3x4<T>::Matrix3x4(T aValues[mSize]) noexcept : mData()
{
    std::copy(aValues, aValues + mSize, mData);
}

// template <RealFloatingPoint T> Matrix3x4<T>::Matrix3x4(const Matrix<T> &aValues3x3, const Vector3<T> &aCol4) noexcept
//{
//     mData[GetIndex(0, 0)] = aValues3x3(0, 0);
//     mData[GetIndex(0, 1)] = aValues3x3(0, 1);
//     mData[GetIndex(0, 2)] = aValues3x3(0, 2);
//     mData[GetIndex(0, 3)] = aCol4.x;
//     mData[GetIndex(1, 0)] = aValues3x3(1, 0);
//     mData[GetIndex(1, 1)] = aValues3x3(1, 1);
//     mData[GetIndex(1, 2)] = aValues3x3(1, 2);
//     mData[GetIndex(1, 3)] = aCol4.y;
//     mData[GetIndex(2, 0)] = aValues3x3(2, 0);
//     mData[GetIndex(2, 1)] = aValues3x3(2, 1);
//     mData[GetIndex(2, 2)] = aValues3x3(2, 2);
//     mData[GetIndex(2, 3)] = aCol4.z;
// }

// template <RealFloatingPoint T>
// Matrix3x4<T>::Matrix3x4(T a00, T a01, T a02, T a03, T a10, T a11, T a12, T a13, T a20, T a21, T a22, T a23) noexcept
//     : mData() // NOLINT
//{
//     mData[GetIndex(0, 0)] = a00;
//     mData[GetIndex(0, 1)] = a01;
//     mData[GetIndex(0, 2)] = a02;
//     mData[GetIndex(0, 3)] = a03;
//     mData[GetIndex(1, 0)] = a10;
//     mData[GetIndex(1, 1)] = a11;
//     mData[GetIndex(1, 2)] = a12;
//     mData[GetIndex(1, 3)] = a13;
//     mData[GetIndex(2, 0)] = a20;
//     mData[GetIndex(2, 1)] = a21;
//     mData[GetIndex(2, 2)] = a22;
//     mData[GetIndex(2, 3)] = a23;
// }

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