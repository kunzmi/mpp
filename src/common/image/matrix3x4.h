#pragma once
#include "../dllexport_common.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/matrix.h>
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
/// A 3x4 matrix for general computation
/// Inner storage order is row major order
/// </summary>
template <RealFloatingPoint T> class MPPEXPORT_COMMON Matrix3x4
{
  private:
    static constexpr int const mCols = 4;
    static constexpr int const mRows = 3;
    static constexpr int const mSize = mCols * mRows;
    T mData[mSize]; // use a standard C-array as we want to use matrix also on GPU
    static constexpr size_t GetIndex(int aRow, int aCol);

  public:
    /// <summary>
    /// Unit matrix (diagonal is 1)
    /// </summary>
    DEVICE_CODE constexpr Matrix3x4() noexcept
    {
        mData[0]  = T(1);
        mData[5]  = T(1);
        mData[10] = T(1);
        mData[1]  = T(0);
        mData[2]  = T(0);
        mData[3]  = T(0);
        mData[4]  = T(0);
        mData[6]  = T(0);
        mData[7]  = T(0);
        mData[8]  = T(0);
        mData[9]  = T(0);
        mData[11] = T(0);
    }

    /// <summary>
    /// All values filled with value aX
    /// </summary>
    DEVICE_CODE constexpr explicit Matrix3x4(T aX) noexcept
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
    }

    /// <summary>
    /// All values filled with aValues
    /// </summary>
    explicit Matrix3x4(const T aValues[mSize]) noexcept;

    /// <summary>
    /// All values filled with aValues
    /// </summary>
    explicit Matrix3x4(const T aValues[mRows][mCols]) noexcept;

    /// <summary>
    /// Set the left 3x3 matrix segment to aValues3x3, set the 4th column to aCol4
    /// </summary>
    DEVICE_CODE constexpr Matrix3x4(const Matrix<T> &aValues3x3, const Vector3<T> &aCol4) noexcept
    {
        mData[0]  = aValues3x3[0];
        mData[1]  = aValues3x3[1];
        mData[2]  = aValues3x3[2];
        mData[3]  = aCol4.x;
        mData[4]  = aValues3x3[3];
        mData[5]  = aValues3x3[4];
        mData[6]  = aValues3x3[5];
        mData[7]  = aCol4.y;
        mData[8]  = aValues3x3[6];
        mData[9]  = aValues3x3[7];
        mData[10] = aValues3x3[8];
        mData[11] = aCol4.z;
    }

    /// <summary>
    /// Creates a new matrix with the given entries:<para/>
    /// a00, a01, a02, a03 <para/>
    /// a10, a11, a12, a13 <para/>
    /// a20, a21, a22, a23
    /// </summary>
    DEVICE_CODE constexpr Matrix3x4(T a00, T a01, T a02, T a03, T a10, T a11, T a12, T a13, T a20, T a21, T a22,
                                    T a23) noexcept // NOLINT
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
    }

    ~Matrix3x4() = default;

    // for conversion
    template <RealFloatingPoint T2>
    explicit Matrix3x4(const Matrix3x4<T2> &aOther) noexcept
        requires(!std::same_as<T, T2>);

    Matrix3x4(const Matrix3x4 &) noexcept            = default;
    Matrix3x4(Matrix3x4 &&) noexcept                 = default;
    Matrix3x4 &operator=(const Matrix3x4 &) noexcept = default; // NOLINT
    Matrix3x4 &operator=(Matrix3x4 &&) noexcept      = default;

    bool operator==(const Matrix3x4 &aOther) const;

    bool operator!=(const Matrix3x4 &aOther) const;

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
    Matrix3x4 &operator+=(T aOther);

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix3x4 &operator+=(const Matrix3x4 &aOther);

    /// <summary>
    /// Element wise addition
    /// </summary>
    Matrix3x4 operator+(const Matrix3x4 &aOther) const;

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix3x4 &operator-=(T aOther);

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix3x4 &operator-=(const Matrix3x4 &aOther);

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    Matrix3x4 operator-(const Matrix3x4 &aOther) const;

    /// <summary>
    /// Element wise multiplication
    /// </summary>
    Matrix3x4 &operator*=(T aOther);

    /// <summary>
    /// Element wise division
    /// </summary>
    Matrix3x4 &operator/=(T aOther);

    friend MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Matrix3x4 &aMat)
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
        return aOs;
        // NOLINTEND
    }
    friend MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Matrix3x4 &aMat)
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
        return aOs;
        // NOLINTEND
    }
    friend MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Matrix3x4 &aMat)
    {
        for (size_t i = 0; i < to_size_t(Matrix3x4::mSize); i++)
        {
            aIs >> aMat[i];
        }
        return aIs;
    }
    friend MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Matrix3x4 &aMat)
    {
        for (size_t i = 0; i < to_size_t(Matrix3x4::mSize); i++)
        {
            aIs >> aMat[i];
        }
        return aIs;
    }
};

template <RealFloatingPoint T> Matrix3x4<T> MPPEXPORT_COMMON operator+(const Matrix3x4<T> &aLeft, T aRight);
template <RealFloatingPoint T> Matrix3x4<T> MPPEXPORT_COMMON operator+(T aLeft, const Matrix3x4<T> &aRight);

template <RealFloatingPoint T> Matrix3x4<T> MPPEXPORT_COMMON operator-(const Matrix3x4<T> &aLeft, T aRight);
template <RealFloatingPoint T> Matrix3x4<T> MPPEXPORT_COMMON operator-(T aLeft, const Matrix3x4<T> &aRight);

template <RealFloatingPoint T> Matrix3x4<T> MPPEXPORT_COMMON operator*(const Matrix3x4<T> &aLeft, T aRight);
template <RealFloatingPoint T> Matrix3x4<T> MPPEXPORT_COMMON operator*(T aLeft, const Matrix3x4<T> &aRight);

template <RealFloatingPoint T> Matrix3x4<T> MPPEXPORT_COMMON operator/(const Matrix3x4<T> &aLeft, T aRight);

/// <summary>
/// Matrix3x4 - vector multiplication <para/>
/// assuming vector is column vector and 4th vector element is 1
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector3<T> operator*(const Matrix3x4<T> &aLeft, const Vector3<T> &aRight)
{
    return Vector3<T>{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z + aLeft[3],
                      aLeft[4] * aRight.x + aLeft[5] * aRight.y + aLeft[6] * aRight.z + aLeft[7],
                      aLeft[8] * aRight.x + aLeft[9] * aRight.y + aLeft[10] * aRight.z + aLeft[11]};
}
/// <summary>
/// Matrix3x4 - vector multiplication <para/>
/// assuming vector is column vector and 4th vector element is 1, ignoring alpha
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector4A<T> operator*(const Matrix3x4<T> &aLeft, const Vector4A<T> &aRight)
{
    return Vector4A<T>{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z + aLeft[3],
                       aLeft[4] * aRight.x + aLeft[5] * aRight.y + aLeft[6] * aRight.z + aLeft[7],
                       aLeft[8] * aRight.x + aLeft[9] * aRight.y + aLeft[10] * aRight.z + aLeft[11]};
}
/// <summary>
/// Matrix3x4 - vector multiplication <para/>
/// assuming vector is column vector and 4th vector element is 1, resulting last element is set to 1
/// </summary>
template <RealFloatingPoint T>
DEVICE_CODE constexpr Vector4<T> operator*(const Matrix3x4<T> &aLeft, const Vector4<T> &aRight)
{
    return Vector4<T>{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z + aLeft[3],
                      aLeft[4] * aRight.x + aLeft[5] * aRight.y + aLeft[6] * aRight.z + aLeft[7],
                      aLeft[8] * aRight.x + aLeft[9] * aRight.y + aLeft[10] * aRight.z + aLeft[11], static_cast<T>(1)};
}

} // namespace mpp::image