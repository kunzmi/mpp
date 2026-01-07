#pragma once
#include "../dllexport_common.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/defines.h>
#include <common/exception.h>
#include <common/numberTypes.h>
#include <common/safeCast.h>
#include <common/utilities.h>
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
/// A 4x4 matrix for general computation
/// Inner storage order is row major order
/// </summary>
template <RealFloatingPoint T> class MPPEXPORT_COMMON Matrix4x4
{
  private:
    static constexpr int const mCols = 4;
    static constexpr int const mRows = 4;
    static constexpr int const mSize = mCols * mRows;
    T mData[mSize]; // use a standard C-array as we want to use matrix also on GPU
    static constexpr size_t GetIndex(int aRow, int aCol);

  public:
    /// <summary>
    /// Unit matrix (diagonal is 1)
    /// </summary>
    DEVICE_CODE constexpr Matrix4x4() noexcept
    {
        mData[0]  = T(1);
        mData[5]  = T(1);
        mData[10] = T(1);
        mData[15] = T(1);
        mData[1]  = T(0);
        mData[2]  = T(0);
        mData[3]  = T(0);
        mData[4]  = T(0);
        mData[6]  = T(0);
        mData[7]  = T(0);
        mData[8]  = T(0);
        mData[9]  = T(0);
        mData[11] = T(0);
        mData[12] = T(0);
        mData[13] = T(0);
        mData[14] = T(0);
    }

    /// <summary>
    /// All values filled with value aX
    /// </summary>
    DEVICE_CODE constexpr explicit Matrix4x4(T aX) noexcept
    {
        mData[0]  = aX;
        mData[1]  = aX;
        mData[2]  = aX;
        mData[3]  = aX;
        mData[4]  = aX;
        mData[5]  = aX;
        mData[6]  = aX;
        mData[7]  = aX;
        mData[8]  = aX;
        mData[9]  = aX;
        mData[10] = aX;
        mData[11] = aX;
        mData[12] = aX;
        mData[13] = aX;
        mData[14] = aX;
        mData[15] = aX;
    }

    /// <summary>
    /// All values filled with aValues
    /// </summary>
    explicit Matrix4x4(T aValues[mSize]) noexcept;

    /// <summary>
    /// Creates a new matrix with the given entries:<para/>
    /// a00, a01, a02, a03 <para/>
    /// a10, a11, a12, a13 <para/>
    /// a20, a21, a22, a23 <para/>
    /// a30, a31, a32, a33
    /// </summary>
    DEVICE_CODE constexpr Matrix4x4(T a00, T a01, T a02, T a03, T a10, T a11, T a12, T a13, T a20, T a21, T a22, T a23,
                                    T a30, T a31, T a32,
                                    T a33) noexcept // NOLINT
    {
        mData[0]  = a00;
        mData[1]  = a01;
        mData[2]  = a02;
        mData[3]  = a03;
        mData[4]  = a10;
        mData[5]  = a11;
        mData[6]  = a12;
        mData[7]  = a13;
        mData[8]  = a20;
        mData[9]  = a21;
        mData[10] = a22;
        mData[11] = a23;
        mData[12] = a30;
        mData[13] = a31;
        mData[14] = a32;
        mData[15] = a33;
    }

    ~Matrix4x4() = default;

    // for conversion
    template <RealFloatingPoint T2>
    explicit Matrix4x4(const Matrix4x4<T2> &aOther) noexcept
        requires(!std::same_as<T, T2>);

    Matrix4x4(const Matrix4x4 &) noexcept            = default;
    Matrix4x4(Matrix4x4 &&) noexcept                 = default;
    Matrix4x4 &operator=(const Matrix4x4 &) noexcept = default; // NOLINT
    Matrix4x4 &operator=(Matrix4x4 &&) noexcept      = default;

    bool operator==(const Matrix4x4 &aOther) const;

    bool operator!=(const Matrix4x4 &aOther) const;

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
    Matrix4x4 &operator+=(T aOther);

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix4x4 &operator+=(const Matrix4x4 &aOther);

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix4x4 operator+(const Matrix4x4 &aOther) const;

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix4x4 &operator-=(T aOther);

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix4x4 &operator-=(const Matrix4x4 &aOther);

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix4x4 operator-(const Matrix4x4 &aOther) const;

    /// <summary>
    /// Element wise multiplication
    /// </summary>
    Matrix4x4 &operator*=(T aOther);

    /// <summary>
    /// Matrix4x4-matrix multiplication
    /// </summary>
    Matrix4x4 &operator*=(const Matrix4x4 &aOther);

    /// <summary>
    /// Matrix4x4-matrix multiplication
    /// </summary>
    DEVICE_CODE constexpr Matrix4x4 operator*(const Matrix4x4 &aOther) const
    {
        Matrix4x4 ret;

        ret.mData[0]  = mData[0] * aOther[0] + mData[1] * aOther[4] + mData[2] * aOther[8] + mData[3] * aOther[12];
        ret.mData[1]  = mData[0] * aOther[1] + mData[1] * aOther[5] + mData[2] * aOther[9] + mData[3] * aOther[13];
        ret.mData[2]  = mData[0] * aOther[2] + mData[1] * aOther[6] + mData[2] * aOther[10] + mData[3] * aOther[14];
        ret.mData[3]  = mData[0] * aOther[3] + mData[1] * aOther[7] + mData[2] * aOther[11] + mData[3] * aOther[15];
        ret.mData[4]  = mData[4] * aOther[0] + mData[5] * aOther[4] + mData[6] * aOther[8] + mData[7] * aOther[12];
        ret.mData[5]  = mData[4] * aOther[1] + mData[5] * aOther[5] + mData[6] * aOther[9] + mData[7] * aOther[13];
        ret.mData[6]  = mData[4] * aOther[2] + mData[5] * aOther[6] + mData[6] * aOther[10] + mData[7] * aOther[14];
        ret.mData[7]  = mData[4] * aOther[3] + mData[5] * aOther[7] + mData[6] * aOther[11] + mData[7] * aOther[15];
        ret.mData[8]  = mData[8] * aOther[0] + mData[9] * aOther[4] + mData[10] * aOther[8] + mData[11] * aOther[12];
        ret.mData[9]  = mData[8] * aOther[1] + mData[9] * aOther[5] + mData[10] * aOther[9] + mData[11] * aOther[13];
        ret.mData[10] = mData[8] * aOther[2] + mData[9] * aOther[6] + mData[10] * aOther[10] + mData[11] * aOther[14];
        ret.mData[11] = mData[8] * aOther[3] + mData[9] * aOther[7] + mData[10] * aOther[11] + mData[11] * aOther[15];
        ret.mData[12] = mData[12] * aOther[0] + mData[13] * aOther[4] + mData[14] * aOther[8] + mData[15] * aOther[12];
        ret.mData[13] = mData[12] * aOther[1] + mData[13] * aOther[5] + mData[14] * aOther[9] + mData[15] * aOther[13];
        ret.mData[14] = mData[12] * aOther[2] + mData[13] * aOther[6] + mData[14] * aOther[10] + mData[15] * aOther[14];
        ret.mData[15] = mData[12] * aOther[3] + mData[13] * aOther[7] + mData[14] * aOther[11] + mData[15] * aOther[15];

        return ret;
    }

    /// <summary>
    /// Element wise division
    /// </summary>
    Matrix4x4 &operator/=(T aOther);

    friend MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Matrix4x4 &aMat)
    {
        std::streamsize maxSize = 0;
        for (const auto &elem : aMat.mData)
        {
            maxSize = std::max(std::streamsize(std::to_string(elem).length()), maxSize);
        }

        // NOLINTBEGIN
        aOs << '(' << std::setw(maxSize) << std::to_string(aMat[0]) << " " << std::setw(maxSize)
            << std::to_string(aMat[1]) << " " << std::setw(maxSize) << std::to_string(aMat[2]) << " "
            << std::setw(maxSize) << std::to_string(aMat[3]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aMat[4]) << " " << std::setw(maxSize)
            << std::to_string(aMat[5]) << " " << std::setw(maxSize) << std::to_string(aMat[6]) << " "
            << std::setw(maxSize) << std::to_string(aMat[7]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aMat[8]) << " " << std::setw(maxSize)
            << std::to_string(aMat[9]) << " " << std::setw(maxSize) << std::to_string(aMat[10]) << " "
            << std::setw(maxSize) << std::to_string(aMat[11]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aMat[12]) << " " << std::setw(maxSize)
            << std::to_string(aMat[13]) << " " << std::setw(maxSize) << std::to_string(aMat[14]) << " "
            << std::setw(maxSize) << std::to_string(aMat[15]) << ')' << std::endl;
        return aOs;
        // NOLINTEND
    }
    friend MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Matrix4x4 &aMat)
    {
        std::streamsize maxSize = 0;
        for (const auto &elem : aMat.mData)
        {
            maxSize = std::max(std::streamsize(std::to_wstring(elem).length()), maxSize);
        }

        // NOLINTBEGIN
        aOs << '(' << std::setw(maxSize) << std::to_wstring(aMat[0]) << " " << std::setw(maxSize)
            << std::to_wstring(aMat[1]) << " " << std::setw(maxSize) << std::to_wstring(aMat[2]) << " "
            << std::setw(maxSize) << std::to_wstring(aMat[3]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_wstring(aMat[4]) << " " << std::setw(maxSize)
            << std::to_wstring(aMat[5]) << " " << std::setw(maxSize) << std::to_wstring(aMat[6]) << " "
            << std::setw(maxSize) << std::to_wstring(aMat[7]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_wstring(aMat[8]) << " " << std::setw(maxSize)
            << std::to_wstring(aMat[9]) << " " << std::setw(maxSize) << std::to_wstring(aMat[10]) << " "
            << std::setw(maxSize) << std::to_wstring(aMat[11]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_wstring(aMat[12]) << " " << std::setw(maxSize)
            << std::to_wstring(aMat[13]) << " " << std::setw(maxSize) << std::to_wstring(aMat[14]) << " "
            << std::setw(maxSize) << std::to_wstring(aMat[15]) << ')' << std::endl;
        return aOs;
        // NOLINTEND
    }
    friend MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Matrix4x4 &aMat)
    {
        for (size_t i = 0; i < to_size_t(Matrix4x4::mSize); i++)
        {
            aIs >> aMat[i];
        }
        return aIs;
    }
    friend MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Matrix4x4 &aMat)
    {
        for (size_t i = 0; i < to_size_t(Matrix4x4::mSize); i++)
        {
            aIs >> aMat[i];
        }
        return aIs;
    }

    /// <summary>
    /// Matrix4x4 inverse
    /// </summary>
    [[nodiscard]] Matrix4x4 Inverse() const;

    /// <summary>
    /// Matrix4x4 transpose
    /// </summary>
    [[nodiscard]] Matrix4x4 Transpose() const;

    /// <summary>
    /// Matrix4x4 determinante
    /// </summary>
    [[nodiscard]] T Det() const;

    /// <summary>
    /// Matrix4x4 trace (sum of the diagonal elements)
    /// </summary>
    [[nodiscard]] T Trace() const;

    /// <summary>
    /// The matrix diagonal elements
    /// </summary>
    [[nodiscard]] Vector4<T> Diagonal() const;
};

template <RealFloatingPoint T> Matrix4x4<T> MPPEXPORT_COMMON operator+(const Matrix4x4<T> &aLeft, T aRight);
template <RealFloatingPoint T> Matrix4x4<T> MPPEXPORT_COMMON operator+(T aLeft, const Matrix4x4<T> &aRight);

template <RealFloatingPoint T> Matrix4x4<T> MPPEXPORT_COMMON operator-(const Matrix4x4<T> &aLeft, T aRight);
template <RealFloatingPoint T> Matrix4x4<T> MPPEXPORT_COMMON operator-(T aLeft, const Matrix4x4<T> &aRight);

template <RealFloatingPoint T> Matrix4x4<T> MPPEXPORT_COMMON operator*(const Matrix4x4<T> &aLeft, T aRight);
template <RealFloatingPoint T> Matrix4x4<T> MPPEXPORT_COMMON operator*(T aLeft, const Matrix4x4<T> &aRight);

template <RealFloatingPoint T> Matrix4x4<T> MPPEXPORT_COMMON operator/(const Matrix4x4<T> &aLeft, T aRight);

/// <summary>
/// Matrix4x4 - vector multiplication <para/>
/// assuming vector is column vector
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector4<T> operator*(const Matrix4x4<T> &aLeft, const Vector4<T> &aRight)
{
    return Vector4<T>{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z + aLeft[3] * aRight.w,
                      aLeft[4] * aRight.x + aLeft[5] * aRight.y + aLeft[6] * aRight.z + aLeft[7] * aRight.w,
                      aLeft[8] * aRight.x + aLeft[9] * aRight.y + aLeft[10] * aRight.z + aLeft[11] * aRight.w,
                      aLeft[12] * aRight.x + aLeft[13] * aRight.y + aLeft[14] * aRight.z + aLeft[15] * aRight.w};
}

/// <summary>
/// Vector-matrix multiplication <para/>
/// assuming vector is row vector
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector4<T> operator*(const Vector4<T> &aLeft, const Matrix4x4<T> &aRight)
{
    return Vector4<T>{aLeft.x * aRight[0] + aLeft.y * aRight[4] + aLeft.z * aRight[8] + aLeft.w * aRight[12],
                      aLeft.x * aRight[1] + aLeft.y * aRight[5] + aLeft.z * aRight[9] + aLeft.w * aRight[13],
                      aLeft.x * aRight[2] + aLeft.y * aRight[6] + aLeft.z * aRight[10] + aLeft.w * aRight[14],
                      aLeft.x * aRight[3] + aLeft.y * aRight[7] + aLeft.z * aRight[11] + aLeft.w * aRight[15]};
}

} // namespace mpp::image