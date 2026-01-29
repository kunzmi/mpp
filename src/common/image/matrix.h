#pragma once
#include "../dllexport_common.h"
#include "affineTransformation.h"
#include "quad.h"
#include "roi.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/solve.h>
#include <common/numberTypes.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector2.h>
#include <common/vector3.h>
#include <common/vector4.h>
#include <common/vector4A.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <ios>
#include <istream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace mpp::image
{
/// <summary>
/// A 3x3 matrix for general computation
/// Inner storage order is row major order
/// </summary>
template <RealFloatingPoint T> class MPPEXPORT_COMMON Matrix
{
  private:
    static constexpr int const mCols = 3;
    static constexpr int const mRows = 3;
    static constexpr int const mSize = mCols * mRows;
    T mData[mSize]; // use a standard C-array as we want to use matrix also on GPU
    static constexpr size_t GetIndex(int aRow, int aCol);

  public:
    /// <summary>
    /// Unit matrix (diagonal is 1)
    /// </summary>
    DEVICE_CODE constexpr Matrix() noexcept
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
    DEVICE_CODE constexpr explicit Matrix(T aX) noexcept
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
    explicit Matrix(const T aValues[mSize]) noexcept;

    /// <summary>
    /// All values filled with aValues
    /// </summary>
    explicit Matrix(const T aValues[mRows][mCols]) noexcept;

    /// <summary>
    /// All values from aAffine and [0 0 1] in last row
    /// </summary>
    explicit Matrix(const AffineTransformation<T> &aAffine) noexcept;

    /// <summary>
    /// Creates a new matrix with the given entries:<para/>
    /// a00, a01, a02 <para/>
    /// a10, a11, a12 <para/>
    /// a20, a21, a22
    /// </summary>
    DEVICE_CODE constexpr Matrix(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22) noexcept // NOLINT
    {
        mData[0] = a00;
        mData[1] = a01;
        mData[2] = a02;
        mData[3] = a10;
        mData[4] = a11;
        mData[5] = a12;
        mData[6] = a20;
        mData[7] = a21;
        mData[8] = a22;
    }

    ~Matrix() = default;

    // for conversion
    template <RealFloatingPoint T2>
    explicit Matrix(const Matrix<T2> &aOther) noexcept
        requires(!std::same_as<T, T2>);

    Matrix(const Matrix &) noexcept            = default;
    Matrix(Matrix &&) noexcept                 = default;
    Matrix &operator=(const Matrix &) noexcept = default; // NOLINT
    Matrix &operator=(Matrix &&) noexcept      = default;

    /// <summary>
    /// Estimates a perspective transformation for four given points
    /// </summary>
    static Matrix FromPoints(const std::pair<Vector2<T>, Vector2<T>> &aP0, const std::pair<Vector2<T>, Vector2<T>> &aP1,
                             const std::pair<Vector2<T>, Vector2<T>> &aP2,
                             const std::pair<Vector2<T>, Vector2<T>> &aP3);

    /// <summary>
    /// Estimates a perspective transformation from aSrcQuad and corresponding coordinates given in aDstQuad
    /// </summary>
    static Matrix FromQuads(const Quad<T> &aSrcQuad, const Quad<T> &aDstQuad);

    bool operator==(const Matrix &aOther) const;

    bool operator!=(const Matrix &aOther) const;

    T &operator()(int aRow, int aCol);
    const T &operator()(int aRow, int aCol) const;

    /// <summary>
    /// Row-major order
    /// </summary>
    T &operator[](int aFlatIndex);

    /// <summary>
    /// Row-major order
    /// </summary>
    DEVICE_CODE constexpr const T &operator[](int aFlatIndex) const
    {
#ifdef IS_HOST_COMPILER
        assert(aFlatIndex >= 0);
        assert(aFlatIndex < mSize);
        return mData[to_size_t(aFlatIndex)]; // NOLINT
#else
        return mData[static_cast<size_t>(aFlatIndex)]; // NOLINT
#endif
    }

    /// <summary>
    /// Row-major order
    /// </summary>
    T &operator[](size_t aFlatIndex);

    /// <summary>
    /// Row-major order
    /// </summary>
    DEVICE_CODE constexpr const T &operator[](size_t aFlatIndex) const
    {
#ifdef IS_HOST_COMPILER
        assert(to_int(aFlatIndex) < mSize);
#endif
        return mData[aFlatIndex]; // NOLINT
    }

    /// <summary>
    /// Pointer to inner data array
    /// </summary>
    [[nodiscard]] const T *Data() const;

    /// <summary>
    /// Pointer to inner data array
    /// </summary>
    [[nodiscard]] T *Data();

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix &operator+=(T aOther);

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix &operator+=(const Matrix &aOther);

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix operator+(const Matrix &aOther) const;

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix &operator-=(T aOther);

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix &operator-=(const Matrix &aOther);

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix operator-(const Matrix &aOther) const;

    /// <summary>
    /// Element wise multiplication
    /// </summary>
    Matrix &operator*=(T aOther);

    /// <summary>
    /// Matrix-matrix multiplication
    /// </summary>
    Matrix &operator*=(const Matrix &aOther);

    /// <summary>
    /// Matrix-matrix multiplication
    /// </summary>
    constexpr DEVICE_CODE Matrix operator*(const Matrix &aOther) const
    {
        Matrix ret;

        ret.mData[0] = mData[0] * aOther[0] + mData[1] * aOther[3] + mData[2] * aOther[6];
        ret.mData[1] = mData[0] * aOther[1] + mData[1] * aOther[4] + mData[2] * aOther[7];
        ret.mData[2] = mData[0] * aOther[2] + mData[1] * aOther[5] + mData[2] * aOther[8];
        ret.mData[3] = mData[3] * aOther[0] + mData[4] * aOther[3] + mData[5] * aOther[6];
        ret.mData[4] = mData[3] * aOther[1] + mData[4] * aOther[4] + mData[5] * aOther[7];
        ret.mData[5] = mData[3] * aOther[2] + mData[4] * aOther[5] + mData[5] * aOther[8];
        ret.mData[6] = mData[6] * aOther[0] + mData[7] * aOther[3] + mData[8] * aOther[6];
        ret.mData[7] = mData[6] * aOther[1] + mData[7] * aOther[4] + mData[8] * aOther[7];
        ret.mData[8] = mData[6] * aOther[2] + mData[7] * aOther[5] + mData[8] * aOther[8];

        return ret;
    }

    /// <summary>
    /// Element wise division
    /// </summary>
    Matrix &operator/=(T aOther);

    friend MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Matrix &aMat)
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
    friend MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Matrix &aMat)
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
    friend MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Matrix &aMat)
    {
        for (size_t i = 0; i < to_size_t(Matrix::mSize); i++)
        {
            aIs >> aMat[i];
        }
        return aIs;
    }
    friend MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Matrix &aMat)
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
    [[nodiscard]] Matrix Inverse() const;

    /// <summary>
    /// Matrix transpose
    /// </summary>
    [[nodiscard]] Matrix Transpose() const;

    /// <summary>
    /// Matrix determinante
    /// </summary>
    [[nodiscard]] T Det() const;

    /// <summary>
    /// Matrix trace (sum of the diagonal elements)
    /// </summary>
    [[nodiscard]] T Trace() const;

    /// <summary>
    /// The matrix diagonal elements
    /// </summary>
    [[nodiscard]] Vector3<T> Diagonal() const;
};

template <RealFloatingPoint T> Matrix<T> MPPEXPORT_COMMON operator+(const Matrix<T> &aLeft, T aRight);
template <RealFloatingPoint T> Matrix<T> MPPEXPORT_COMMON operator+(T aLeft, const Matrix<T> &aRight);

template <RealFloatingPoint T> Matrix<T> MPPEXPORT_COMMON operator-(const Matrix<T> &aLeft, T aRight);
template <RealFloatingPoint T> Matrix<T> MPPEXPORT_COMMON operator-(T aLeft, const Matrix<T> &aRight);

template <RealFloatingPoint T> Matrix<T> MPPEXPORT_COMMON operator*(const Matrix<T> &aLeft, T aRight);
template <RealFloatingPoint T> Matrix<T> MPPEXPORT_COMMON operator*(T aLeft, const Matrix<T> &aRight);

template <RealFloatingPoint T> Matrix<T> MPPEXPORT_COMMON operator/(const Matrix<T> &aLeft, T aRight);

/// <summary>
/// Matrix - vector multiplication <para/>
/// assuming vector is column vector
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector3<T> operator*(const Matrix<T> &aLeft, const Vector3<T> &aRight)
{
    return Vector3<T>{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z,
                      aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5] * aRight.z,
                      aLeft[6] * aRight.x + aLeft[7] * aRight.y + aLeft[8] * aRight.z};
}

/// <summary>
/// Vector-matrix multiplication <para/>
/// assuming vector is row vector
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector3<T> operator*(const Vector3<T> &aLeft, const Matrix<T> &aRight)
{
    return Vector3<T>{aLeft.x * aRight[0] + aLeft.y * aRight[3] + aLeft.z * aRight[6],
                      aLeft.x * aRight[1] + aLeft.y * aRight[4] + aLeft.z * aRight[7],
                      aLeft.x * aRight[2] + aLeft.y * aRight[5] + aLeft.z * aRight[8]};
}

/// <summary>
/// Matrix - vector multiplication <para/>
/// assuming vector is column vector
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector4A<T> operator*(const Matrix<T> &aLeft, const Vector4A<T> &aRight)
{
    return Vector4A<T>{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z,
                       aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5] * aRight.z,
                       aLeft[6] * aRight.x + aLeft[7] * aRight.y + aLeft[8] * aRight.z};
}

/// <summary>
/// Vector-matrix multiplication <para/>
/// assuming vector is row vector
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector4A<T> operator*(const Vector4A<T> &aLeft, const Matrix<T> &aRight)
{
    return Vector4A<T>{aLeft.x * aRight[0] + aLeft.y * aRight[3] + aLeft.z * aRight[6],
                       aLeft.x * aRight[1] + aLeft.y * aRight[4] + aLeft.z * aRight[7],
                       aLeft.x * aRight[2] + aLeft.y * aRight[5] + aLeft.z * aRight[8]};
}

/// <summary>
/// Matrix - vector multiplication <para/>
/// assuming vector is column vector - last element is set to 1
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector4<T> operator*(const Matrix<T> &aLeft, const Vector4<T> &aRight)
{
    return Vector4<T>{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z,
                      aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5] * aRight.z,
                      aLeft[6] * aRight.x + aLeft[7] * aRight.y + aLeft[8] * aRight.z, static_cast<T>(1)};
}

/// <summary>
/// Transform every point in a Quad
/// </summary>
template <RealFloatingPoint T> Quad<T> MPPEXPORT_COMMON operator*(const Matrix<T> &aLeft, const Quad<T> &aRight);

/// <summary>
/// Transform every corner point of a ROI
/// </summary>
template <RealFloatingPoint T> Quad<T> MPPEXPORT_COMMON operator*(const Matrix<T> &aLeft, const Roi &aRight);

/// <summary>
/// perspective transformation matrix - vector multiplication <para/>
/// assuming vector is column vector and z element is 1
/// </summary>
template <RealFloatingPoint T> DEVICE_CODE Vector2<T> operator*(const Matrix<T> &aLeft, const Vector2<T> &aRight)
{
    const T scaling = aLeft[6] * aRight.x + aLeft[7] * aRight.y + aLeft[8];
    // don't throw any exception in case scaling == 0, as they don't exist on GPU
    return Vector2<T>{(aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2]) / scaling,
                      (aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5]) / scaling};
}

template <RealFloatingPoint T> using PerspectiveTransformation = Matrix<T>;
} // namespace mpp::image