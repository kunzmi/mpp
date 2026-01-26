#include "../dllexport_common.h"
#include "affineTransformation.h"
#include "matrix.h"
#include "quad.h"
#include "roi.h"
#include <algorithm>
#include <cassert>
#include <common/image/matrixException.h>
#include <common/image/solve.h>
#include <common/numberTypes.h>
#include <common/safeCast.h>
#include <common/vector2.h>
#include <common/vector3.h>
#include <concepts>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace mpp::image
{
template <RealFloatingPoint T> constexpr size_t Matrix<T>::GetIndex(int aRow, int aCol)
{
    assert(aRow >= 0);
    assert(aCol >= 0);
    assert(aRow < mRows);
    assert(aCol < mCols);

    return to_size_t(aCol + aRow * mCols);
}

template <RealFloatingPoint T> Matrix<T>::Matrix(const T aValues[mSize]) noexcept : mData()
{
    std::copy(aValues, aValues + mSize, mData);
}

template <RealFloatingPoint T> Matrix<T>::Matrix(const T aValues[mRows][mCols]) noexcept : mData()
{
    std::copy(reinterpret_cast<const T *>(aValues), reinterpret_cast<const T *>(aValues) + mSize, mData);
}

template <RealFloatingPoint T> Matrix<T>::Matrix(const AffineTransformation<T> &aAffine) noexcept : mData()
{
    mData[0] = aAffine.Data()[0];
    mData[1] = aAffine.Data()[1];
    mData[2] = aAffine.Data()[2];
    mData[3] = aAffine.Data()[3];
    mData[4] = aAffine.Data()[4];
    mData[5] = aAffine.Data()[5];
    mData[6] = T(0);
    mData[7] = T(0);
    mData[8] = T(1);
}

template <RealFloatingPoint T>
Matrix<T> Matrix<T>::FromPoints(const std::pair<Vector2<T>, Vector2<T>> &aP0,
                                const std::pair<Vector2<T>, Vector2<T>> &aP1,
                                const std::pair<Vector2<T>, Vector2<T>> &aP2,
                                const std::pair<Vector2<T>, Vector2<T>> &aP3)
{
    const Vector2<T> xy0 = aP0.first;
    const Vector2<T> xy1 = aP1.first;
    const Vector2<T> xy2 = aP2.first;
    const Vector2<T> xy3 = aP3.first;

    const Vector2<T> uv0 = aP0.second;
    const Vector2<T> uv1 = aP1.second;
    const Vector2<T> uv2 = aP2.second;
    const Vector2<T> uv3 = aP3.second;

    /*
     * Coefficients are calculated by solving linear system:
     * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
     * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
     * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
     * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
     * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
     * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
     * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
     * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
     */
    std::vector<T> matrix = {
        xy0.x, xy0.y, 1, 0,     0,     0, -xy0.x * uv0.x, -xy0.y * uv0.x, uv0.x, //
        xy1.x, xy1.y, 1, 0,     0,     0, -xy1.x * uv1.x, -xy1.y * uv1.x, uv1.x, //
        xy2.x, xy2.y, 1, 0,     0,     0, -xy2.x * uv2.x, -xy2.y * uv2.x, uv2.x, //
        xy3.x, xy3.y, 1, 0,     0,     0, -xy3.x * uv3.x, -xy3.y * uv3.x, uv3.x, //
        0,     0,     0, xy0.x, xy0.y, 1, -xy0.x * uv0.y, -xy0.y * uv0.y, uv0.y, //
        0,     0,     0, xy1.x, xy1.y, 1, -xy1.x * uv1.y, -xy1.y * uv1.y, uv1.y, //
        0,     0,     0, xy2.x, xy2.y, 1, -xy2.x * uv2.y, -xy2.y * uv2.y, uv2.y, //
        0,     0,     0, xy3.x, xy3.y, 1, -xy3.x * uv3.y, -xy3.y * uv3.y, uv3.y  //
    };

    Matrix<T> ret;
    if (!solve(matrix.data(), ret.mData, 8))
    {
        throw MATRIXEXCEPTION("Failed to solve linear system of equations.");
    }

    // the last matrix entry is not calculated and must be set to 1:
    ret.mData[8] = 1;
    return ret;
}

template <RealFloatingPoint T> Matrix<T> Matrix<T>::FromQuads(const Quad<T> &aSrcQuad, const Quad<T> &aDstQuad)
{
    const Vector2<T> xy0 = aSrcQuad.P0;
    const Vector2<T> xy1 = aSrcQuad.P1;
    const Vector2<T> xy2 = aSrcQuad.P2;
    const Vector2<T> xy3 = aSrcQuad.P3;

    const Vector2<T> uv0 = aDstQuad.P0;
    const Vector2<T> uv1 = aDstQuad.P1;
    const Vector2<T> uv2 = aDstQuad.P2;
    const Vector2<T> uv3 = aDstQuad.P3;

    /*
     * Coefficients are calculated by solving linear system:
     * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
     * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
     * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
     * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
     * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
     * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
     * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
     * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
     */
    std::vector<T> matrix = {
        xy0.x, xy0.y, 1, 0,     0,     0, -xy0.x * uv0.x, -xy0.y * uv0.x, uv0.x, //
        xy1.x, xy1.y, 1, 0,     0,     0, -xy1.x * uv1.x, -xy1.y * uv1.x, uv1.x, //
        xy2.x, xy2.y, 1, 0,     0,     0, -xy2.x * uv2.x, -xy2.y * uv2.x, uv2.x, //
        xy3.x, xy3.y, 1, 0,     0,     0, -xy3.x * uv3.x, -xy3.y * uv3.x, uv3.x, //
        0,     0,     0, xy0.x, xy0.y, 1, -xy0.x * uv0.y, -xy0.y * uv0.y, uv0.y, //
        0,     0,     0, xy1.x, xy1.y, 1, -xy1.x * uv1.y, -xy1.y * uv1.y, uv1.y, //
        0,     0,     0, xy2.x, xy2.y, 1, -xy2.x * uv2.y, -xy2.y * uv2.y, uv2.y, //
        0,     0,     0, xy3.x, xy3.y, 1, -xy3.x * uv3.y, -xy3.y * uv3.y, uv3.y  //
    };

    Matrix<T> ret;
    if (!solve(matrix.data(), ret.mData, 8))
    {
        throw MATRIXEXCEPTION("Failed to solve linear system of equations.");
    }

    // the last matrix entry is not calculated and must be set to 1:
    ret.mData[8] = 1;
    return ret;
}

// for conversion
template <RealFloatingPoint T>
template <RealFloatingPoint T2>
Matrix<T>::Matrix(const Matrix<T2> &aOther) noexcept
    requires(!std::same_as<T, T2>)
{
    for (size_t i = 0; i < mSize; i++)
    {
        mData[i] = static_cast<T>(aOther.Data()[i]); // NOLINT
    }
}

template <RealFloatingPoint T> bool Matrix<T>::operator==(const Matrix &aOther) const
{
    bool ret = true;
    for (size_t i = 0; i < mSize; i++)
    {
        ret &= mData[i] == aOther[i]; // NOLINT
    }
    return ret;
}

template <RealFloatingPoint T> bool Matrix<T>::operator!=(const Matrix &aOther) const
{
    return !(*this == aOther);
}

template <RealFloatingPoint T> T &Matrix<T>::operator()(int aRow, int aCol)
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT
}
template <RealFloatingPoint T> const T &Matrix<T>::operator()(int aRow, int aCol) const
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT
}

template <RealFloatingPoint T> T &Matrix<T>::operator[](int aFlatIndex)
{
    assert(aFlatIndex >= 0);
    assert(aFlatIndex < mSize);
    return mData[to_size_t(aFlatIndex)]; // NOLINT
}

template <RealFloatingPoint T> T &Matrix<T>::operator[](size_t aFlatIndex)
{
    assert(to_int(aFlatIndex) < mSize);
    return mData[aFlatIndex]; // NOLINT
}

template <RealFloatingPoint T> const T *Matrix<T>::Data() const
{
    return mData;
}

template <RealFloatingPoint T> T *Matrix<T>::Data()
{
    return mData;
}

template <RealFloatingPoint T> Matrix<T> &Matrix<T>::operator+=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN += aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix<T> &Matrix<T>::operator+=(const Matrix &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::plus<>{});
    return *this;
}

template <RealFloatingPoint T> Matrix<T> Matrix<T>::operator+(const Matrix &aOther) const
{
    Matrix ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::plus<>{});
    return ret;
}

template <RealFloatingPoint T> Matrix<T> &Matrix<T>::operator-=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN -= aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix<T> &Matrix<T>::operator-=(const Matrix &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::minus<>{});
    return *this;
}

template <RealFloatingPoint T> Matrix<T> Matrix<T>::operator-(const Matrix &aOther) const
{
    Matrix ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::minus<>{});
    return ret;
}

template <RealFloatingPoint T> Matrix<T> &Matrix<T>::operator*=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN *= aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix<T> &Matrix<T>::operator*=(const Matrix &aOther)
{
    const Matrix ret = *this * aOther;
    *this            = ret;
    return *this;
}

template <RealFloatingPoint T> Matrix<T> &Matrix<T>::operator/=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN /= aOther; });
    return *this;
}

template <RealFloatingPoint T> Matrix<T> Matrix<T>::Inverse() const
{
    const T det = Det();

    if (det == 0)
    {
        throw MATRIXEXCEPTION("Cannot compute Matrix inverse as determinant is 0.");
    }

    const T invdet = static_cast<T>(1) / det;

    Matrix ret;
    ret[0] = (mData[4] * mData[8] - mData[7] * mData[5]) * invdet;
    ret[1] = (mData[2] * mData[7] - mData[1] * mData[8]) * invdet;
    ret[2] = (mData[1] * mData[5] - mData[2] * mData[4]) * invdet;
    ret[3] = (mData[5] * mData[6] - mData[3] * mData[8]) * invdet;
    ret[4] = (mData[0] * mData[8] - mData[2] * mData[6]) * invdet;
    ret[5] = (mData[3] * mData[2] - mData[0] * mData[5]) * invdet;
    ret[6] = (mData[3] * mData[7] - mData[6] * mData[4]) * invdet;
    ret[7] = (mData[6] * mData[1] - mData[0] * mData[7]) * invdet;
    ret[8] = (mData[0] * mData[4] - mData[3] * mData[1]) * invdet;
    return ret;
}

template <RealFloatingPoint T> Matrix<T> Matrix<T>::Transpose() const
{
    Matrix ret(*this);
    ret[1] = mData[3];
    ret[3] = mData[1];

    ret[2] = mData[6];
    ret[6] = mData[2];

    ret[5] = mData[7];
    ret[7] = mData[5];
    return ret;
}

template <RealFloatingPoint T> T Matrix<T>::Det() const
{
    return mData[0] * (mData[4] * mData[8] - mData[5] * mData[7]) -
           mData[1] * (mData[3] * mData[8] - mData[5] * mData[6]) +
           mData[2] * (mData[3] * mData[7] - mData[4] * mData[6]);
}

template <RealFloatingPoint T> T Matrix<T>::Trace() const
{
    return mData[0] + mData[4] + mData[8];
}

template <RealFloatingPoint T> Vector3<T> Matrix<T>::Diagonal() const
{
    return {mData[0], mData[4], mData[8]};
}

template <RealFloatingPoint T> Matrix<T> operator+(const Matrix<T> &aLeft, T aRight)
{
    Matrix ret(aLeft);
    ret += aRight;
    return ret;
}
template <RealFloatingPoint T> Matrix<T> operator+(T aLeft, const Matrix<T> &aRight)
{
    Matrix ret(aLeft);
    ret += aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix<T> operator-(const Matrix<T> &aLeft, T aRight)
{
    Matrix ret(aLeft);
    ret -= aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix<T> operator-(T aLeft, const Matrix<T> &aRight)
{
    Matrix ret(aLeft);
    ret -= aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix<T> operator*(const Matrix<T> &aLeft, T aRight)
{
    Matrix ret(aLeft);
    ret *= aRight;
    return ret;
}

template <RealFloatingPoint T> Matrix<T> operator*(T aLeft, const Matrix<T> &aRight)
{
    Matrix ret(aRight);
    ret *= aLeft;
    return ret;
}

template <RealFloatingPoint T> Matrix<T> operator/(const Matrix<T> &aLeft, T aRight)
{
    Matrix ret(aLeft);
    ret *= T(1) / aRight;
    return ret;
}

template <RealFloatingPoint T> Quad<T> operator*(const Matrix<T> &aLeft, const Quad<T> &aRight)
{
    return {aLeft * aRight.P0, aLeft * aRight.P1, aLeft * aRight.P2, aLeft * aRight.P3};
}

template <RealFloatingPoint T> Quad<T> operator*(const Matrix<T> &aLeft, const Roi &aRight)
{
    return aLeft * Quad<T>(aRight);
}

// instantiate for float and double:
template class Matrix<float>;
template class Matrix<double>;

template MPPEXPORT_COMMON Matrix<float>::Matrix(const Matrix<double> &aOther);
template MPPEXPORT_COMMON Matrix<double>::Matrix(const Matrix<float> &aOther);

template Matrix<float> MPPEXPORT_COMMON operator+(const Matrix<float> &aLeft, float aRight);
template Matrix<float> MPPEXPORT_COMMON operator+(float aLeft, const Matrix<float> &aRight);
template Matrix<float> MPPEXPORT_COMMON operator-(const Matrix<float> &aLeft, float aRight);
template Matrix<float> MPPEXPORT_COMMON operator-(float aLeft, const Matrix<float> &aRight);

template Matrix<float> MPPEXPORT_COMMON operator*(const Matrix<float> &aLeft, float aRight);
template Matrix<float> MPPEXPORT_COMMON operator*(float aLeft, const Matrix<float> &aRight);
template Matrix<float> MPPEXPORT_COMMON operator/(const Matrix<float> &aLeft, float aRight);

template Quad<float> MPPEXPORT_COMMON operator*(const Matrix<float> &aLeft, const Quad<float> &aRight);
template Quad<float> MPPEXPORT_COMMON operator*(const Matrix<float> &aLeft, const Roi &aRight);

template Matrix<double> MPPEXPORT_COMMON operator+(const Matrix<double> &aLeft, double aRight);
template Matrix<double> MPPEXPORT_COMMON operator+(double aLeft, const Matrix<double> &aRight);
template Matrix<double> MPPEXPORT_COMMON operator-(const Matrix<double> &aLeft, double aRight);
template Matrix<double> MPPEXPORT_COMMON operator-(double aLeft, const Matrix<double> &aRight);

template Matrix<double> MPPEXPORT_COMMON operator*(const Matrix<double> &aLeft, double aRight);
template Matrix<double> MPPEXPORT_COMMON operator*(double aLeft, const Matrix<double> &aRight);
template Matrix<double> MPPEXPORT_COMMON operator/(const Matrix<double> &aLeft, double aRight);

template Quad<double> MPPEXPORT_COMMON operator*(const Matrix<double> &aLeft, const Quad<double> &aRight);
template Quad<double> MPPEXPORT_COMMON operator*(const Matrix<double> &aLeft, const Roi &aRight);

} // namespace mpp::image