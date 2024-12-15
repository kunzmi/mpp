#pragma once
#include "affineTransformation.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/solve.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector2.h>
#include <common/vector3.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <ios>
#include <istream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace opp::image
{
// forward declaration:
class AffineTransformation;

/// <summary>
/// A 3x3 matrix for general computation
/// Inner storage order is row major order
/// </summary>
template <FloatingPoint T> class Matrix
{
  private:
    static constexpr int const mCols = 3;
    static constexpr int const mRows = 3;
    static constexpr int const mSize = mCols * mRows;
    T mData[mSize]; // use a standard C-array as we want to use matrix also on GPU
    static constexpr size_t GetIndex(int aRow, int aCol)
    {
        assert(aRow >= 0);
        assert(aCol >= 0);
        assert(aRow < mRows);
        assert(aCol < mCols);

        return to_size_t(aCol + aRow * mCols);
    }

  public:
    /// <summary>
    /// Unit matrix (diagonal is 1)
    /// </summary>
    Matrix() noexcept : mData()
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

    /// <summary>
    /// All values filled with value aX
    /// </summary>
    explicit Matrix(T aX) noexcept : mData()
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

    /// <summary>
    /// All values filled with aValues
    /// </summary>
    explicit Matrix(T aValues[mSize]) noexcept : mData()
    {
        std::copy(aValues, aValues + mSize, mData);
    }

    /// <summary>
    /// All values from aAffine and [0 0 1] in last row
    /// </summary>
    explicit Matrix(const AffineTransformation &aAffine) noexcept : mData()
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

    /// <summary>
    /// Creates a new matrix with the given entries:<para/>
    /// a00, a01, a02 <para/>
    /// a10, a11, a12 <para/>
    /// a20, a21, a22
    /// </summary>
    Matrix(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22) noexcept : mData() // NOLINT
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

    /// <summary>
    /// Estimates an perspective transformation for four given points
    /// </summary>
    Matrix(const std::pair<Vec2d, Vec2d> &aP0, const std::pair<Vec2d, Vec2d> &aP1, const std::pair<Vec2d, Vec2d> &aP2,
           const std::pair<Vec2d, Vec2d> &aP3) noexcept
        requires std::same_as<T, double>
        : mData()
    {
        bool ok = false;
        try
        {
            const Vec2d xy0 = aP0.first;
            const Vec2d xy1 = aP1.first;
            const Vec2d xy2 = aP2.first;
            const Vec2d xy3 = aP3.first;

            const Vec2d uv0 = aP0.second;
            const Vec2d uv1 = aP1.second;
            const Vec2d uv2 = aP2.second;
            const Vec2d uv3 = aP3.second;

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
            std::vector<double> matrix = {
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
        mData[8] = 1.0;

        if (!ok) // failed to solve system of equation
        {
            // set to unit matrix
            mData[0] = 1.0;
            mData[4] = 1.0;
            mData[8] = 1.0;
            mData[1] = 0.0;
            mData[2] = 0.0;
            mData[3] = 0.0;
            mData[5] = 0.0;
            mData[6] = 0.0;
            mData[7] = 0.0;
        }
    }

    ~Matrix() = default;

    Matrix(const Matrix &) noexcept            = default;
    Matrix(Matrix &&) noexcept                 = default;
    Matrix &operator=(const Matrix &) noexcept = default;
    Matrix &operator=(Matrix &&) noexcept      = default;

    bool operator==(const Matrix &aOther) const
    {
        bool ret = true;
        for (size_t i = 0; i < mSize; i++)
        {
            ret &= mData[i] == aOther[i]; // NOLINT
        }
        return ret;
    }

    bool operator!=(const Matrix &aOther) const
    {
        return !(*this == aOther);
    }

    T &operator()(int aRow, int aCol)
    {
        return mData[GetIndex(aRow, aCol)];
    }
    T const &operator()(int aRow, int aCol) const
    {
        return mData[GetIndex(aRow, aCol)];
    }

    /// <summary>
    /// Row-major order
    /// </summary>
    T &operator[](int aFlatIndex)
    {
        assert(aFlatIndex >= 0);
        assert(aFlatIndex < mSize);
        return mData[to_size_t(aFlatIndex)]; // NOLINT
    }

    /// <summary>
    /// Row-major order
    /// </summary>
    T const &operator[](int aFlatIndex) const
    {
        assert(aFlatIndex >= 0);
        assert(aFlatIndex < mSize);
        return mData[to_size_t(aFlatIndex)]; // NOLINT
    }

    /// <summary>
    /// Row-major order
    /// </summary>
    T &operator[](size_t aFlatIndex)
    {
        assert(to_int(aFlatIndex) < mSize);
        return mData[aFlatIndex]; // NOLINT
    }

    /// <summary>
    /// Row-major order
    /// </summary>
    T const &operator[](size_t aFlatIndex) const
    {
        assert(to_int(aFlatIndex) < mSize);
        return mData[aFlatIndex]; // NOLINT
    }

    /// <summary>
    /// Pointer to inner data array
    /// </summary>
    [[nodiscard]] const T *Data() const
    {
        return mData;
    }

    /// <summary>
    /// Pointer to inner data array
    /// </summary>
    [[nodiscard]] T *Data()
    {
        return mData;
    }

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix &operator+=(T aOther)
    {
        std::for_each(mData, mData + mSize, [&](auto &aN) { aN += aOther; });
        return *this;
    }

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix &operator+=(const Matrix &aOther)
    {
        std::transform(mData, mData + mSize, aOther.mData, mData, std::plus<>{});
        return *this;
    }

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix operator+(const Matrix &aOther) const
    {
        Matrix ret;
        std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::plus<>{});
        return ret;
    }

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix &operator-=(T aOther)
    {
        std::for_each(mData, mData + mSize, [&](auto &aN) { aN -= aOther; });
        return *this;
    }

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix &operator-=(const Matrix &aOther)
    {
        std::transform(mData, mData + mSize, aOther.mData, mData, std::minus<>{});
        return *this;
    }

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix operator-(const Matrix &aOther) const
    {
        Matrix ret;
        std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::minus<>{});
        return ret;
    }

    /// <summary>
    /// Element wise multiplication
    /// </summary>
    Matrix &operator*=(T aOther)
    {
        std::for_each(mData, mData + mSize, [&](auto &aN) { aN *= aOther; });
        return *this;
    }

    /// <summary>
    /// Matrix-matrix multiplication
    /// </summary>
    Matrix &operator*=(const Matrix &aOther)
    {
        const Matrix ret = *this * aOther;
        *this            = ret;
        return *this;
    }

    /// <summary>
    /// Matrix-matrix multiplication
    /// </summary>
    Matrix operator*(const Matrix &aOther) const
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

    /// <summary>
    /// Element wise division
    /// </summary>
    Matrix &operator/=(T aOther)
    {
        std::for_each(mData, mData + mSize, [&](auto &aN) { aN /= aOther; });
        return *this;
    }

    friend std::ostream &operator<<(std::ostream &aOs, const Matrix &aMat)
    {
        std::streamsize maxSize = 0;
        for (const auto &elem : aMat.mData)
        {
            maxSize = std::max(std::streamsize(std::to_string(elem).length()), maxSize);
        }

        // NOLINTBEGIN
        aOs << '(' << std::setw(maxSize) << std::to_string(aMat[0]) << " " << std::setw(maxSize)
            << std::to_string(aMat[1]) << " " << std::setw(maxSize) << std::to_string(aMat[2]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aMat[3]) << " " << std::setw(maxSize)
            << std::to_string(aMat[4]) << " " << std::setw(maxSize) << std::to_string(aMat[5]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aMat[6]) << " " << std::setw(maxSize)
            << std::to_string(aMat[7]) << " " << std::setw(maxSize) << std::to_string(aMat[8]) << ')' << std::endl;
        return aOs;
        // NOLINTEND
    }
    friend std::wostream &operator<<(std::wostream &aOs, const Matrix &aMat)
    {
        std::streamsize maxSize = 0;
        for (const auto &elem : aMat.mData)
        {
            maxSize = std::max(std::streamsize(std::to_wstring(elem).length()), maxSize);
        }

        // NOLINTBEGIN
        aOs << '(' << std::setw(maxSize) << std::to_wstring(aMat[0]) << " " << std::setw(maxSize)
            << std::to_wstring(aMat[1]) << " " << std::setw(maxSize) << std::to_wstring(aMat[2]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_wstring(aMat[3]) << " " << std::setw(maxSize)
            << std::to_wstring(aMat[4]) << " " << std::setw(maxSize) << std::to_wstring(aMat[5]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_wstring(aMat[6]) << " " << std::setw(maxSize)
            << std::to_wstring(aMat[7]) << " " << std::setw(maxSize) << std::to_wstring(aMat[8]) << ')' << std::endl;
        return aOs;
        // NOLINTEND
    }
    friend std::istream &operator>>(std::istream &aIs, Matrix &aMat)
    {
        for (size_t i = 0; i < to_size_t(Matrix::mSize); i++)
        {
            aIs >> aMat[i];
        }
        return aIs;
    }
    friend std::wistream &operator>>(std::wistream &aIs, Matrix &aMat)
    {
        for (size_t i = 0; i < to_size_t(Matrix::mSize); i++)
        {
            aIs >> aMat[i];
        }
        return aIs;
    }

    /// <summary>
    /// Matrix inverse
    /// </summary>
    [[nodiscard]] Matrix Inverse() const
    {
        const double det = Det();

        if (det == 0)
        {
            throw EXCEPTION("Cannot compute Matrix inverse as determinant is 0.");
        }

        const double invdet = 1.0 / det;

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

    /// <summary>
    /// Matrix transpose
    /// </summary>
    [[nodiscard]] Matrix Transpose() const
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

    /// <summary>
    /// Matrix determinante
    /// </summary>
    [[nodiscard]] T Det() const
    {
        return mData[0] * (mData[4] * mData[8] - mData[5] * mData[7]) -
               mData[1] * (mData[3] * mData[8] - mData[5] * mData[6]) +
               mData[2] * (mData[3] * mData[7] - mData[4] * mData[6]);
    }

    /// <summary>
    /// Matrix trace (sum of the diagonal elements)
    /// </summary>
    [[nodiscard]] T Trace() const
    {
        return mData[0] + mData[4] + mData[8];
    }

    /// <summary>
    /// The matrix diagonal elements
    /// </summary>
    [[nodiscard]] Vector3<T> Diagonal() const
    {
        return {mData[0], mData[4], mData[8]};
    }
};

template <FloatingPoint T> Matrix<T> operator+(const Matrix<T> &aLeft, T aRight)
{
    Matrix ret(aLeft);
    ret += aRight;
    return ret;
}
template <FloatingPoint T> Matrix<T> operator+(T aLeft, const Matrix<T> &aRight)
{
    Matrix ret(aLeft);
    ret += aRight;
    return ret;
}

template <FloatingPoint T> Matrix<T> operator-(const Matrix<T> &aLeft, T aRight)
{
    Matrix ret(aLeft);
    ret -= aRight;
    return ret;
}

template <FloatingPoint T> Matrix<T> operator-(T aLeft, const Matrix<T> &aRight)
{
    Matrix ret(aLeft);
    ret -= aRight;
    return ret;
}

template <FloatingPoint T> Matrix<T> operator*(const Matrix<T> &aLeft, T aRight)
{
    Matrix ret(aLeft);
    ret *= aRight;
    return ret;
}

template <FloatingPoint T> Matrix<T> operator*(T aLeft, const Matrix<T> &aRight)
{
    Matrix ret(aRight);
    ret *= aLeft;
    return ret;
}

template <FloatingPoint T> Matrix<T> operator/(const Matrix<T> &aLeft, T aRight)
{
    Matrix ret(aLeft);
    ret *= T(1) / aRight;
    return ret;
}

/// <summary>
/// Matrix - vector multiplication <para/>
/// assuming vector is column vector
/// </summary>
template <FloatingPoint T> Vector3<T> operator*(const Matrix<T> &aLeft, const Vector3<T> &aRight)
{
    return Vector3<T>{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z,
                      aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5] * aRight.z,
                      aLeft[6] * aRight.x + aLeft[7] * aRight.y + aLeft[8] * aRight.z};
}

/// <summary>
/// Vector-matrix multiplication <para/>
/// assuming vector is row vector
/// </summary>
template <FloatingPoint T> Vector3<T> operator*(const Vector3<T> &aLeft, const Matrix<T> &aRight)
{
    return Vector3<T>{aLeft.x * aRight[0] + aLeft.y * aRight[3] + aLeft.z * aRight[6],
                      aLeft.x * aRight[1] + aLeft.y * aRight[4] + aLeft.z * aRight[7],
                      aLeft.x * aRight[2] + aLeft.y * aRight[5] + aLeft.z * aRight[8]};
}

/// <summary>
/// perspective transformation matrix - vector multiplication <para/>
/// assuming vector is column vector and z element is 1
/// </summary>
template <FloatingPoint T> Vector2<T> operator*(const Matrix<T> &aLeft, const Vector2<T> &aRight)
{
    const T scaling = aLeft[6] * aRight.x + aLeft[7] * aRight.y + aLeft[8];
    // don't throw any exception in case scaling == 0, as they don't exist on GPU
    return Vector2<T>{(aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2]) / scaling,
                      (aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5]) / scaling};
}

// some common color space conversion matrices (values as used in NPP)
extern const Matrix<float> RGBtoYUV;
extern const Matrix<float> YUVtoRGB;

extern const Matrix<float> RGBtoYCbCr;
extern const Matrix<float> CbCrtoRGB;

extern const Matrix<float> RGBtoXYZ;
extern const Matrix<float> XYZtoRGB;
} // namespace opp::image