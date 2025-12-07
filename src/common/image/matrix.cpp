#include "../dllexport_common.h"
#include "affineTransformation.h"
#include "matrix.h"
#include "quad.h"
#include "roi.h"
#include <algorithm>
#include <cassert>
#include <common/exception.h>
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

template <RealFloatingPoint T> Matrix<T>::Matrix() noexcept : mData()
{
    mData[0] = T(1);
    mData[4] = T(1);
    mData[8] = T(1);
    mData[1] = T(0);
    mData[2] = T(0);
    mData[3] = T(0);
    mData[5] = T(0);
    mData[6] = T(0);
    mData[7] = T(0);
}

template <RealFloatingPoint T> Matrix<T>::Matrix(T aX) noexcept : mData()
{
    mData[0] = aX;
    mData[1] = aX;
    mData[2] = aX;
    mData[3] = aX;
    mData[4] = aX;
    mData[5] = aX;
    mData[6] = aX;
    mData[7] = aX;
    mData[8] = aX;
}

template <RealFloatingPoint T> Matrix<T>::Matrix(T aValues[mSize]) noexcept : mData()
{
    std::copy(aValues, aValues + mSize, mData);
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
Matrix<T>::Matrix(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22) noexcept : mData() // NOLINT
{
    mData[GetIndex(0, 0)] = a00;
    mData[GetIndex(0, 1)] = a01;
    mData[GetIndex(0, 2)] = a02;
    mData[GetIndex(1, 0)] = a10;
    mData[GetIndex(1, 1)] = a11;
    mData[GetIndex(1, 2)] = a12;
    mData[GetIndex(2, 0)] = a20;
    mData[GetIndex(2, 1)] = a21;
    mData[GetIndex(2, 2)] = a22;
}

template <RealFloatingPoint T>
Matrix<T>::Matrix(const std::pair<Vector2<T>, Vector2<T>> &aP0, const std::pair<Vector2<T>, Vector2<T>> &aP1,
                  const std::pair<Vector2<T>, Vector2<T>> &aP2, const std::pair<Vector2<T>, Vector2<T>> &aP3) noexcept
    : mData()
{
    bool ok = false;
    try
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

        ok = solve(matrix.data(), mData, 8);
    }
    catch (...)
    {
        ok = false;
    }

    // the last matrix entry is not calculated and must be set to 1:
    mData[8] = 1;

    if (!ok) // failed to solve system of equation
    {
        // set to unit matrix
        mData[0] = 1;
        mData[4] = 1;
        mData[8] = 1;
        mData[1] = 0;
        mData[2] = 0;
        mData[3] = 0;
        mData[5] = 0;
        mData[6] = 0;
        mData[7] = 0;
    }
}

template <RealFloatingPoint T>
Matrix<T>::Matrix(const Roi &aRoi, const Quad<T> &aQuad) noexcept : Matrix(Quad<T>(aRoi), aQuad)
{
}

template <RealFloatingPoint T> Matrix<T>::Matrix(const Quad<T> &aSrcQuad, const Quad<T> &aDstQuad) noexcept : mData()
{
    bool ok = false;
    try
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

        ok = solve(matrix.data(), mData, 8);
    }
    catch (...)
    {
        ok = false;
    }

    // the last matrix entry is not calculated and must be set to 1:
    mData[8] = 1;

    if (!ok) // failed to solve system of equation
    {
        // set to unit matrix
        mData[0] = 1;
        mData[4] = 1;
        mData[8] = 1;
        mData[1] = 0;
        mData[2] = 0;
        mData[3] = 0;
        mData[5] = 0;
        mData[6] = 0;
        mData[7] = 0;
    }
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

// template <RealFloatingPoint T> const T &Matrix<T>::operator[](int aFlatIndex) const
//{
//     assert(aFlatIndex >= 0);
//     assert(aFlatIndex < mSize);
//     return mData[to_size_t(aFlatIndex)]; // NOLINT
// }

template <RealFloatingPoint T> T &Matrix<T>::operator[](size_t aFlatIndex)
{
    assert(to_int(aFlatIndex) < mSize);
    return mData[aFlatIndex]; // NOLINT
}

// template <RealFloatingPoint T> const T &Matrix<T>::operator[](size_t aFlatIndex) const
//{
//     assert(to_int(aFlatIndex) < mSize);
//     return mData[aFlatIndex]; // NOLINT
// }

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

template <RealFloatingPoint T> Matrix<T> Matrix<T>::operator*(const Matrix &aOther) const
{
    Matrix ret;

    ret[0] = mData[0] * aOther[0] + mData[1] * aOther[3] + mData[2] * aOther[6];
    ret[1] = mData[0] * aOther[1] + mData[1] * aOther[4] + mData[2] * aOther[7];
    ret[2] = mData[0] * aOther[2] + mData[1] * aOther[5] + mData[2] * aOther[8];
    ret[3] = mData[3] * aOther[0] + mData[4] * aOther[3] + mData[5] * aOther[6];
    ret[4] = mData[3] * aOther[1] + mData[4] * aOther[4] + mData[5] * aOther[7];
    ret[5] = mData[3] * aOther[2] + mData[4] * aOther[5] + mData[5] * aOther[8];
    ret[6] = mData[6] * aOther[0] + mData[7] * aOther[3] + mData[8] * aOther[6];
    ret[7] = mData[6] * aOther[1] + mData[7] * aOther[4] + mData[8] * aOther[7];
    ret[8] = mData[6] * aOther[2] + mData[7] * aOther[5] + mData[8] * aOther[8];

    return ret;
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
        throw EXCEPTION("Cannot compute Matrix inverse as determinant is 0.");
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

template Matrix<float> operator+(const Matrix<float> &aLeft, float aRight);
template Matrix<float> operator+(float aLeft, const Matrix<float> &aRight);
template Matrix<float> operator-(const Matrix<float> &aLeft, float aRight);
template Matrix<float> operator-(float aLeft, const Matrix<float> &aRight);

template Matrix<float> operator*(const Matrix<float> &aLeft, float aRight);
template Matrix<float> operator*(float aLeft, const Matrix<float> &aRight);
template Matrix<float> operator/(const Matrix<float> &aLeft, float aRight);

template Quad<float> operator*(const Matrix<float> &aLeft, const Quad<float> &aRight);
template Quad<float> operator*(const Matrix<float> &aLeft, const Roi &aRight);

template Matrix<double> operator+(const Matrix<double> &aLeft, double aRight);
template Matrix<double> operator+(double aLeft, const Matrix<double> &aRight);
template Matrix<double> operator-(const Matrix<double> &aLeft, double aRight);
template Matrix<double> operator-(double aLeft, const Matrix<double> &aRight);

template Matrix<double> operator*(const Matrix<double> &aLeft, double aRight);
template Matrix<double> operator*(double aLeft, const Matrix<double> &aRight);
template Matrix<double> operator/(const Matrix<double> &aLeft, double aRight);

template Quad<double> operator*(const Matrix<double> &aLeft, const Quad<double> &aRight);
template Quad<double> operator*(const Matrix<double> &aLeft, const Roi &aRight);

const Matrix<float> RGBtoYUV =
    Matrix<float>(0.299f, 0.587f, 0.114f, -0.147f, -0.289f, 0.436f, 0.615f, -0.515f, -0.100f);
const Matrix<float> YUVtoRGB = Matrix<float>(1.0f, 0.0f, 1.140f, 1.0f, -0.394f, -0.581f, 1.0f, 2.032f, 0.0f);

const Matrix<float> RGBtoYCbCr =
    Matrix<float>(0.257f, 0.504f, 0.098f, -0.148f, -0.291f, +0.439f, 0.439f, -0.368f, -0.071f);
const Matrix<float> CbCrtoRGB = Matrix<float>(1.164f, 0.0f, 1.596f, 1.164f, -0.392f, -0.813f, 1.164f, 2.017f, 0.0f);

const Matrix<float> RGBtoXYZ =
    Matrix<float>(0.412453f, 0.35758f, 0.180423f, 0.212671f, 0.71516f, 0.072169f, 0.019334f, 0.119193f, 0.950227f);
const Matrix<float> XYZtoRGB =
    Matrix<float>(3.240479f, -1.53715f, -0.498535f, -0.969256f, 1.875991f, 0.041556f, 0.055648f, -0.204043f, 1.057311f);
} // namespace mpp::image