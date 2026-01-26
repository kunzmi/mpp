#pragma once
#include "../dllexport_common.h"
#include <common/defines.h>
#include <common/image/quad.h>
#include <common/image/roi.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <iomanip>
#include <ios>
#include <istream>
#include <ostream>
#include <string>
#include <utility>

namespace mpp::image
{
/// <summary>
/// A 3x2 AffineTransformation that represents a 2D affine transformation.<para/>
/// The last row of the matrix is assumend to be [0 0 1] and is not explicitly stored.<para/>
/// Inner storage order is row major order.
/// </summary>
template <RealFloatingPoint T> class MPPEXPORT_COMMON AffineTransformation
{
  private:
    static constexpr int const mCols = 3;
    static constexpr int const mRows = 2;
    static constexpr int const mSize = mCols * mRows;
    T mData[mSize]; // use a standard C-array as we want to use AffineTransformation also on GPU
    static constexpr size_t GetIndex(int aRow, int aCol);

  public:
    /// <summary>
    /// Unit AffineTransformation (diagonal is 1)
    /// </summary>
    AffineTransformation() noexcept;

    /// <summary>
    /// All values filled with value aX
    /// </summary>
    explicit AffineTransformation(T aX) noexcept;

    /// <summary>
    /// All values filled with aValues
    /// </summary>
    explicit AffineTransformation(const T aValues[mSize]) noexcept;

    /// <summary>
    /// All values filled with aValues
    /// </summary>
    explicit AffineTransformation(const T aValues[mRows][mCols]) noexcept;

    /// <summary>
    /// Creates a new AffineTransformation with the given entries:<para/>
    /// a00, a01, a02 <para/>
    /// a10, a11, a12 <para/>
    ///   0,   0,   1
    /// </summary>
    AffineTransformation(T a00, T a01, T a02, T a10, T a11, T a12) noexcept;

    ~AffineTransformation() = default;

    // for conversion
    template <RealFloatingPoint T2>
    explicit AffineTransformation(const AffineTransformation<T2> &aOther) noexcept
        requires(!std::same_as<T, T2>);

    // in the following lines, clang-tidy detects 't' in "default" as a magic number :(

    AffineTransformation(const AffineTransformation &) noexcept            = default;
    AffineTransformation(AffineTransformation &&) noexcept                 = default;
    AffineTransformation &operator=(const AffineTransformation &) noexcept = default; // NOLINT
    AffineTransformation &operator=(AffineTransformation &&) noexcept      = default;

    /// <summary>
    /// Estimates an affine transformation for three given points
    /// </summary>
    static AffineTransformation FromPoints(const std::pair<Vector2<T>, Vector2<T>> &aP0,
                                           const std::pair<Vector2<T>, Vector2<T>> &aP1,
                                           const std::pair<Vector2<T>, Vector2<T>> &aP2);

    /// <summary>
    /// Estimates an affine transformation from aSrcQuad to corresponding coordinates given in aDstQuad (fourth
    /// component is ignored)
    /// </summary>
    static AffineTransformation FromQuads(const Quad<T> &aSrcQuad, const Quad<T> &aDstQuad);

    bool operator==(const AffineTransformation &aOther) const;
    bool operator!=(const AffineTransformation &aOther) const;

    T &operator()(int aRow, int aCol);
    T const &operator()(int aRow, int aCol) const;

    /// <summary>
    /// Row-major order
    /// </summary>
    T &operator[](int aFlatIndex);

    /// <summary>
    /// Row-major order
    /// </summary>
    DEVICE_CODE const T &operator[](int aFlatIndex) const
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
    DEVICE_CODE T const &operator[](size_t aFlatIndex) const
    {
#ifdef IS_HOST_COMPILER
        assert(to_int(aFlatIndex) < mSize);
#endif
        return mData[aFlatIndex]; // NOLINT --> non constant array index
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
    AffineTransformation &operator+=(T aOther);

    /// <summary>
    /// Element wise addition
    /// </summary>
    AffineTransformation &operator+=(const AffineTransformation &aOther);

    /// <summary>
    /// Element wise addition
    /// </summary>
    AffineTransformation operator+(const AffineTransformation &aOther) const;

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    AffineTransformation &operator-=(T aOther);

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    AffineTransformation &operator-=(const AffineTransformation &aOther);

    /// <summary>
    /// Element wise subtraction
    /// </summary>
    AffineTransformation operator-(const AffineTransformation &aOther) const;

    /// <summary>
    /// Element wise multiplication
    /// </summary>
    AffineTransformation &operator*=(T aOther);

    /// <summary>
    /// Matrix-Matrix multiplication
    /// </summary>
    AffineTransformation &operator*=(const AffineTransformation &aOther);

    /// <summary>
    /// Matrix-Matrix multiplication
    /// </summary>
    AffineTransformation operator*(const AffineTransformation &aOther) const;

    /// <summary>
    /// Element wise division
    /// </summary>
    AffineTransformation &operator/=(T aOther);

    friend std::ostream &operator<<(std::ostream &aOs, const AffineTransformation<T> &aMat)
    {
        std::streamsize maxSize = 0;
        for (const auto &elem : aMat.mData)
        {
            maxSize = std::max(std::streamsize(std::to_string(elem).length()), maxSize);
        }

        // clang tidy gets a bit crazy with std::streamsize...
        // NOLINTBEGIN
        aOs << '(' << std::setw(maxSize) << std::to_string(aMat[0]) << " " << std::setw(maxSize)
            << std::to_string(aMat[1]) << " " << std::setw(maxSize) << std::to_string(aMat[2]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(aMat[3]) << " " << std::setw(maxSize)
            << std::to_string(aMat[4]) << " " << std::setw(maxSize) << std::to_string(aMat[5]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_string(0.0) << " " << std::setw(maxSize) << std::to_string(0.0)
            << " " << std::setw(maxSize) << std::to_string(1.0) << ')' << std::endl;
        return aOs;
        // NOLINTEND
    }
    friend std::wostream &operator<<(std::wostream &aOs, const AffineTransformation<T> &aMat)
    {
        std::streamsize maxSize = 0;
        for (const auto &elem : aMat.mData)
        {
            maxSize = std::max(std::streamsize(std::to_wstring(elem).length()), maxSize);
        }

        // clang tidy gets a bit crazy with std::streamsize...
        // NOLINTBEGIN
        aOs << '(' << std::setw(maxSize) << std::to_wstring(aMat[0]) << " " << std::setw(maxSize)
            << std::to_wstring(aMat[1]) << " " << std::setw(maxSize) << std::to_wstring(aMat[2]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_wstring(aMat[3]) << " " << std::setw(maxSize)
            << std::to_wstring(aMat[4]) << " " << std::setw(maxSize) << std::to_wstring(aMat[5]) << ')' << std::endl;
        aOs << '(' << std::setw(maxSize) << std::to_wstring(0.0) << " " << std::setw(maxSize) << std::to_wstring(0.0)
            << " " << std::setw(maxSize) << std::to_wstring(1.0) << ')' << std::endl;
        return aOs;
        // NOLINTEND
    }
    friend MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, AffineTransformation<T> &aMat)
    {
        for (size_t i = 0; i < to_size_t(AffineTransformation<T>::mSize); i++)
        {
            aIs >> aMat[i];
        }
        return aIs;
    }
    friend MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, AffineTransformation<T> &aMat)
    {
        for (size_t i = 0; i < to_size_t(AffineTransformation<T>::mSize); i++)
        {
            aIs >> aMat[i];
        }
        return aIs;
    }

    /// <summary>
    /// AffineTransformation inverse
    /// </summary>
    [[nodiscard]] AffineTransformation Inverse() const;

    /// <summary>
    /// AffineTransformation determinante
    /// </summary>
    [[nodiscard]] T Det() const;

    /// <summary>
    /// AffineTransformation trace (sum of the diagonal elements)
    /// </summary>
    [[nodiscard]] T Trace() const;

    /// <summary>
    /// The AffineTransformation diagonal elements
    /// </summary>
    [[nodiscard]] Vector3<T> Diagonal() const;

    /// <summary>
    /// AffineTransformation for image rotation (counter-clock wise)
    /// </summary>
    [[nodiscard]] static AffineTransformation GetRotation(T aAngleInDeg) noexcept;
    /// <summary>
    /// AffineTransformation for image scaling
    /// </summary>
    [[nodiscard]] static AffineTransformation GetScale(T aScale) noexcept;
    /// <summary>
    /// AffineTransformation for image scaling (different for x and y)
    /// </summary>
    [[nodiscard]] static AffineTransformation GetScale(const Vector2<T> &aScale) noexcept;
    /// <summary>
    /// AffineTransformation for image translation
    /// </summary>
    [[nodiscard]] static AffineTransformation GetTranslation(const Vector2<T> &aShift) noexcept;
    /// <summary>
    /// AffineTransformation for image shearing
    /// </summary>
    [[nodiscard]] static AffineTransformation GetShear(const Vector2<T> &aShear) noexcept;
};

template <RealFloatingPoint T>
AffineTransformation<T> MPPEXPORT_COMMON operator+(const AffineTransformation<T> &aLeft, T aRight);
template <RealFloatingPoint T>
AffineTransformation<T> MPPEXPORT_COMMON operator+(T aLeft, const AffineTransformation<T> &aRight);
template <RealFloatingPoint T>
AffineTransformation<T> MPPEXPORT_COMMON operator-(const AffineTransformation<T> &aLeft, T aRight);
template <RealFloatingPoint T>
AffineTransformation<T> MPPEXPORT_COMMON operator-(T aLeft, const AffineTransformation<T> &aRight);

template <RealFloatingPoint T>
AffineTransformation<T> MPPEXPORT_COMMON operator*(const AffineTransformation<T> &aLeft, T aRight);
template <RealFloatingPoint T>
AffineTransformation<T> MPPEXPORT_COMMON operator*(T aLeft, const AffineTransformation<T> &aRight);
template <RealFloatingPoint T>
AffineTransformation<T> MPPEXPORT_COMMON operator/(const AffineTransformation<T> &aLeft, T aRight);

/// <summary>
/// Matrix - vector multiplication <para/>
/// assuming vector is column vector
/// </summary>
template <RealFloatingPoint T>
Vector3<T> MPPEXPORT_COMMON operator*(const AffineTransformation<T> &aLeft, const Vector3<T> &aRight);

/// <summary>
/// Vector-Matrix multiplication <para/>
/// assuming vector is row vector
/// </summary>
template <RealFloatingPoint T>
Vector3<T> MPPEXPORT_COMMON operator*(const Vector3<T> &aLeft, const AffineTransformation<T> &aRight);

/// <summary>
/// Transform every point in a Quad
/// </summary>
template <RealFloatingPoint T>
Quad<T> MPPEXPORT_COMMON operator*(const AffineTransformation<T> &aLeft, const Quad<T> &aRight);

/// <summary>
/// Transform every corner point of a ROI
/// </summary>
template <RealFloatingPoint T>
Quad<T> MPPEXPORT_COMMON operator*(const AffineTransformation<T> &aLeft, const Roi &aRight);

/// <summary>
/// affine matrix - vector multiplication <para/>
/// assuming vector is column vector and z element is 1<para/>
/// Implemented in header so that we have this operator also on GPU
/// </summary>
DEVICE_CODE inline Vec2d operator*(const AffineTransformation<double> &aLeft, const Vec2d &aRight)
{
    return Vec2d{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2],
                 aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5]};
}

/// <summary>
/// affine matrix - vector multiplication <para/>
/// assuming vector is column vector and z element is 1<para/>
/// Implemented in header so that we have this operator also on GPU
/// </summary>
DEVICE_CODE inline Vec2f operator*(const AffineTransformation<float> &aLeft, const Vec2f &aRight)
{
    return Vec2f{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2],
                 aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5]};
}
} // namespace mpp::image