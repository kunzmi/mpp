#include "../dllexport_common.h"
#include "matrix4x4.h"
#include <algorithm>
#include <cassert>
#include <common/exception.h>
#include <common/numberTypes.h>
#include <common/safeCast.h>
#include <common/vector3.h>
#include <common/vector4.h>
#include <concepts>
#include <cstddef>
#include <functional>

namespace mpp::image
{
template <RealFloatingPoint T> constexpr size_t Matrix4x4<T>::GetIndex(int aRow, int aCol)
{
    assert(aRow >= 0);
    assert(aCol >= 0);
    assert(aRow < mRows);
    assert(aCol < mCols);

    return to_size_t(aCol + aRow * mCols);
}

template <RealFloatingPoint T> Matrix4x4<T>::Matrix4x4(const T aValues[mSize]) noexcept : mData()
{
    std::copy(aValues, aValues + mSize, mData);
}

template <RealFloatingPoint T> Matrix4x4<T>::Matrix4x4(const T aValues[mRows][mCols]) noexcept : mData()
{
    std::copy(reinterpret_cast<const T *>(aValues), reinterpret_cast<const T *>(aValues) + mSize, mData);
}

// for conversion
template <RealFloatingPoint T>
template <RealFloatingPoint T2>
Matrix4x4<T>::Matrix4x4(const Matrix4x4<T2> &aOther) noexcept
    requires(!std::same_as<T, T2>)
{
    for (size_t i = 0; i < mSize; i++)
    {
        mData[i] = static_cast<T>(aOther.Data()[i]); // NOLINT
    }
}

template <RealFloatingPoint T> bool Matrix4x4<T>::operator==(const Matrix4x4 &aOther) const
{
    bool ret = true;
    for (size_t i = 0; i < mSize; i++)
    {
        ret &= mData[i] == aOther[i]; // NOLINT
    }
    return ret;
}

template <RealFloatingPoint T> bool Matrix4x4<T>::operator!=(const Matrix4x4 &aOther) const
{
    return !(*this == aOther);
}

template <RealFloatingPoint T> T &Matrix4x4<T>::operator()(int aRow, int aCol)
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT
}
template <RealFloatingPoint T> const T &Matrix4x4<T>::operator()(int aRow, int aCol) const
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT
}

template <RealFloatingPoint T> T &Matrix4x4<T>::operator[](int aFlatIndex)
{
    assert(aFlatIndex >= 0);
    assert(aFlatIndex < mSize);
    return mData[to_size_t(aFlatIndex)]; // NOLINT
}

template <RealFloatingPoint T> T &Matrix4x4<T>::operator[](size_t aFlatIndex)
{
    assert(to_int(aFlatIndex) < mSize);
    return mData[aFlatIndex]; // NOLINT
}

template <RealFloatingPoint T> const T *Matrix4x4<T>::Data() const
{
    return mData;
}

template <RealFloatingPoint T> T *Matrix4x4<T>::Data()
{
    return mData;
}

template <RealFloatingPoint T> Matrix4x4<T> &Matrix4x4<T>::operator+=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN += aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix4x4<T> &Matrix4x4<T>::operator+=(const Matrix4x4 &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::plus<>{});
    return *this;
}

template <RealFloatingPoint T> Matrix4x4<T> Matrix4x4<T>::operator+(const Matrix4x4 &aOther) const
{
    Matrix4x4 ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::plus<>{});
    return ret;
}

template <RealFloatingPoint T> Matrix4x4<T> &Matrix4x4<T>::operator-=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN -= aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix4x4<T> &Matrix4x4<T>::operator-=(const Matrix4x4 &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::minus<>{});
    return *this;
}

template <RealFloatingPoint T> Matrix4x4<T> Matrix4x4<T>::operator-(const Matrix4x4 &aOther) const
{
    Matrix4x4 ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::minus<>{});
    return ret;
}

template <RealFloatingPoint T> Matrix4x4<T> &Matrix4x4<T>::operator*=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN *= aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix4x4<T> &Matrix4x4<T>::operator*=(const Matrix4x4 &aOther)
{
    const Matrix4x4 ret = *this * aOther;
    *this               = ret;
    return *this;
}

// template <RealFloatingPoint T> Matrix4x4<T> Matrix4x4<T>::operator*(const Matrix4x4 &aOther) const
//{
//     Matrix4x4 ret;
//
//     ret.mData[0] = mData[0] * aOther[0] + mData[1] * aOther[4] + mData[2] * aOther[8] + mData[3] * aOther[12];
//     ret.mData[1]  = mData[0] * aOther[1] + mData[1] * aOther[5] + mData[2] * aOther[9] + mData[3] * aOther[13];
//     ret.mData[2]  = mData[0] * aOther[2] + mData[1] * aOther[6] + mData[2] * aOther[10] + mData[3] * aOther[14];
//     ret.mData[3]  = mData[0] * aOther[3] + mData[1] * aOther[7] + mData[2] * aOther[11] + mData[3] * aOther[15];
//     ret.mData[4]  = mData[4] * aOther[0] + mData[5] * aOther[4] + mData[6] * aOther[8] + mData[7] * aOther[12];
//     ret.mData[5]  = mData[4] * aOther[1] + mData[5] * aOther[5] + mData[6] * aOther[9] + mData[7] * aOther[13];
//     ret.mData[6]  = mData[4] * aOther[2] + mData[5] * aOther[6] + mData[6] * aOther[10] + mData[7] * aOther[14];
//     ret.mData[7]  = mData[4] * aOther[3] + mData[5] * aOther[7] + mData[6] * aOther[11] + mData[7] * aOther[15];
//     ret.mData[8]  = mData[8] * aOther[0] + mData[9] * aOther[4] + mData[10] * aOther[8] + mData[11] * aOther[12];
//     ret.mData[9]  = mData[8] * aOther[1] + mData[9] * aOther[5] + mData[10] * aOther[9] + mData[11] * aOther[13];
//     ret.mData[10] = mData[8] * aOther[2] + mData[9] * aOther[6] + mData[10] * aOther[10] + mData[11] * aOther[14];
//     ret.mData[11] = mData[8] * aOther[3] + mData[9] * aOther[7] + mData[10] * aOther[11] + mData[11] * aOther[15];
//     ret.mData[12] = mData[12] * aOther[0] + mData[13] * aOther[4] + mData[14] * aOther[8] + mData[15] * aOther[12];
//     ret.mData[13] = mData[12] * aOther[1] + mData[13] * aOther[5] + mData[14] * aOther[9] + mData[15] * aOther[13];
//     ret.mData[14] = mData[12] * aOther[2] + mData[13] * aOther[6] + mData[14] * aOther[10] + mData[15] * aOther[14];
//     ret.mData[15] = mData[12] * aOther[3] + mData[13] * aOther[7] + mData[14] * aOther[11] + mData[15] * aOther[15];
//
//     return ret;
// }

template <RealFloatingPoint T> Matrix4x4<T> &Matrix4x4<T>::operator/=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN /= aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix4x4<T> Matrix4x4<T>::Inverse() const
{
    const T fA0 = mData[0] * mData[5] - mData[1] * mData[4];
    const T fA1 = mData[0] * mData[6] - mData[2] * mData[4];
    const T fA2 = mData[0] * mData[7] - mData[3] * mData[4];
    const T fA3 = mData[1] * mData[6] - mData[2] * mData[5];
    const T fA4 = mData[1] * mData[7] - mData[3] * mData[5];
    const T fA5 = mData[2] * mData[7] - mData[3] * mData[6];
    const T fB0 = mData[8] * mData[13] - mData[9] * mData[12];
    const T fB1 = mData[8] * mData[14] - mData[10] * mData[12];
    const T fB2 = mData[8] * mData[15] - mData[11] * mData[12];
    const T fB3 = mData[9] * mData[14] - mData[10] * mData[13];
    const T fB4 = mData[9] * mData[15] - mData[11] * mData[13];
    const T fB5 = mData[10] * mData[15] - mData[11] * mData[14];

    const T det = fA0 * fB5 - fA1 * fB4 + fA2 * fB3 + fA3 * fB2 - fA4 * fB1 + fA5 * fB0;

    if (det == 0)
    {
        throw EXCEPTION("Cannot compute Matrix4x4 inverse as determinant is 0.");
    }

    const T invdet = static_cast<T>(1) / det;

    Matrix4x4 ret;
    ret[0]  = +mData[5] * fB5 - mData[6] * fB4 + mData[7] * fB3;
    ret[1]  = -mData[1] * fB5 + mData[2] * fB4 - mData[3] * fB3;
    ret[2]  = +mData[13] * fA5 - mData[14] * fA4 + mData[15] * fA3;
    ret[3]  = -mData[9] * fA5 + mData[10] * fA4 - mData[11] * fA3;
    ret[4]  = -mData[4] * fB5 + mData[6] * fB2 - mData[7] * fB1;
    ret[5]  = +mData[0] * fB5 - mData[2] * fB2 + mData[3] * fB1;
    ret[6]  = -mData[12] * fA5 + mData[14] * fA2 - mData[15] * fA1;
    ret[7]  = +mData[8] * fA5 - mData[10] * fA2 + mData[11] * fA1;
    ret[8]  = +mData[4] * fB4 - mData[5] * fB2 + mData[7] * fB0;
    ret[9]  = -mData[0] * fB4 + mData[1] * fB2 - mData[3] * fB0;
    ret[10] = +mData[12] * fA4 - mData[13] * fA2 + mData[15] * fA0;
    ret[11] = -mData[8] * fA4 + mData[9] * fA2 - mData[11] * fA0;
    ret[12] = -mData[4] * fB3 + mData[5] * fB1 - mData[6] * fB0;
    ret[13] = +mData[0] * fB3 - mData[1] * fB1 + mData[2] * fB0;
    ret[14] = -mData[12] * fA3 + mData[13] * fA1 - mData[14] * fA0;
    ret[15] = +mData[8] * fA3 - mData[9] * fA1 + mData[10] * fA0;

    ret *= invdet;
    return ret;
}

template <RealFloatingPoint T> Matrix4x4<T> Matrix4x4<T>::Transpose() const
{
    return Matrix4x4(mData[0], mData[4], mData[8], mData[12], mData[1], mData[5], mData[9], mData[13], mData[2],
                     mData[6], mData[10], mData[14], mData[3], mData[7], mData[11], mData[15]);
}

template <RealFloatingPoint T> T Matrix4x4<T>::Det() const
{
    const T fA0 = mData[0] * mData[5] - mData[1] * mData[4];
    const T fA1 = mData[0] * mData[6] - mData[2] * mData[4];
    const T fA2 = mData[0] * mData[7] - mData[3] * mData[4];
    const T fA3 = mData[1] * mData[6] - mData[2] * mData[5];
    const T fA4 = mData[1] * mData[7] - mData[3] * mData[5];
    const T fA5 = mData[2] * mData[7] - mData[3] * mData[6];
    const T fB0 = mData[8] * mData[13] - mData[9] * mData[12];
    const T fB1 = mData[8] * mData[14] - mData[10] * mData[12];
    const T fB2 = mData[8] * mData[15] - mData[11] * mData[12];
    const T fB3 = mData[9] * mData[14] - mData[10] * mData[13];
    const T fB4 = mData[9] * mData[15] - mData[11] * mData[13];
    const T fB5 = mData[10] * mData[15] - mData[11] * mData[14];

    return fA0 * fB5 - fA1 * fB4 + fA2 * fB3 + fA3 * fB2 - fA4 * fB1 + fA5 * fB0;
}

template <RealFloatingPoint T> T Matrix4x4<T>::Trace() const
{
    return mData[0] + mData[5] + mData[10] + mData[15];
}

template <RealFloatingPoint T> Vector4<T> Matrix4x4<T>::Diagonal() const
{
    return {mData[0], mData[5], mData[10], mData[15]};
}

template <RealFloatingPoint T> Matrix4x4<T> operator+(const Matrix4x4<T> &aLeft, T aRight)
{
    Matrix4x4 ret(aLeft);
    ret += aRight;
    return ret;
}
template <RealFloatingPoint T> Matrix4x4<T> operator+(T aLeft, const Matrix4x4<T> &aRight)
{
    Matrix4x4 ret(aLeft);
    ret += aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix4x4<T> operator-(const Matrix4x4<T> &aLeft, T aRight)
{
    Matrix4x4 ret(aLeft);
    ret -= aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix4x4<T> operator-(T aLeft, const Matrix4x4<T> &aRight)
{
    Matrix4x4 ret(aLeft);
    ret -= aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix4x4<T> operator*(const Matrix4x4<T> &aLeft, T aRight)
{
    Matrix4x4 ret(aLeft);
    ret *= aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix4x4<T> operator*(T aLeft, const Matrix4x4<T> &aRight)
{
    Matrix4x4 ret(aRight);
    ret *= aLeft;
    return ret;
}

template <RealFloatingPoint T> Matrix4x4<T> operator/(const Matrix4x4<T> &aLeft, T aRight)
{
    Matrix4x4 ret(aLeft);
    ret *= T(1) / aRight;
    return ret;
}

// instantiate for float and double:
template class Matrix4x4<float>;
template class Matrix4x4<double>;

template MPPEXPORT_COMMON Matrix4x4<float>::Matrix4x4(const Matrix4x4<double> &aOther);
template MPPEXPORT_COMMON Matrix4x4<double>::Matrix4x4(const Matrix4x4<float> &aOther);

template Matrix4x4<float> operator+(const Matrix4x4<float> &aLeft, float aRight);
template Matrix4x4<float> operator+(float aLeft, const Matrix4x4<float> &aRight);
template Matrix4x4<float> operator-(const Matrix4x4<float> &aLeft, float aRight);
template Matrix4x4<float> operator-(float aLeft, const Matrix4x4<float> &aRight);

template Matrix4x4<float> operator*(const Matrix4x4<float> &aLeft, float aRight);
template Matrix4x4<float> operator*(float aLeft, const Matrix4x4<float> &aRight);
template Matrix4x4<float> operator/(const Matrix4x4<float> &aLeft, float aRight);

template Matrix4x4<double> operator+(const Matrix4x4<double> &aLeft, double aRight);
template Matrix4x4<double> operator+(double aLeft, const Matrix4x4<double> &aRight);
template Matrix4x4<double> operator-(const Matrix4x4<double> &aLeft, double aRight);
template Matrix4x4<double> operator-(double aLeft, const Matrix4x4<double> &aRight);

template Matrix4x4<double> operator*(const Matrix4x4<double> &aLeft, double aRight);
template Matrix4x4<double> operator*(double aLeft, const Matrix4x4<double> &aRight);
template Matrix4x4<double> operator/(const Matrix4x4<double> &aLeft, double aRight);

} // namespace mpp::image