#pragma once
#include <common/vectorTypes.h>
#include <istream>
#include <ostream>
#include <utility>

namespace opp::image
{
/// <summary>
/// A 3x2 AffineTransformation that represents a 2D affine transformation.<para/>
/// The last row is assumend to be [0 0 1]. <para/>
/// Inner storage order is row major order
/// </summary>
class AffineTransformation
{
  private:
    static constexpr int const mCols = 3;
    static constexpr int const mRows = 2;
    static constexpr int const mSize = mCols * mRows;
    double mData[mSize]; // use a standard C-array as we want to use AffineTransformation also on GPU
    static constexpr size_t GetIndex(int aRow, int aCol);

  public:
    /// <summary>
    /// Unit AffineTransformation (diagonal is 1)
    /// </summary>
    AffineTransformation() noexcept;

    /// <summary>
    /// All values filled with value aX
    /// </summary>
    explicit AffineTransformation(double aX) noexcept;

    /// <summary>
    /// All values filled with aValues
    /// </summary>
    explicit AffineTransformation(double aValues[mSize]) noexcept;

    /// <summary>
    /// Creates a new AffineTransformation with the given entries:<para/>
    /// a00, a01, a02 <para/>
    /// a10, a11, a12 <para/>
    ///   0,   0,   1
    /// </summary>
    AffineTransformation(double a00, double a01, double a02, double a10, double a11, double a12) noexcept;

    /// <summary>
    /// Estimates an affine transformation for three given points
    /// </summary>
    AffineTransformation(const std::pair<Vec2d, Vec2d> &aP0, const std::pair<Vec2d, Vec2d> &aP1,
                         const std::pair<Vec2d, Vec2d> &aP2) noexcept;

    ~AffineTransformation() = default;

    // in the following lines, clang-tidy detects 't' in "default" as a magic number :(

    AffineTransformation(const AffineTransformation &) noexcept            = default;
    AffineTransformation(AffineTransformation &&) noexcept                 = default;
    AffineTransformation &operator=(const AffineTransformation &) noexcept = default; // NOLINT
    AffineTransformation &operator=(AffineTransformation &&) noexcept      = default;

    bool operator==(const AffineTransformation &aOther) const;
    bool operator!=(const AffineTransformation &aOther) const;

    double &operator()(int aRow, int aCol);
    double const &operator()(int aRow, int aCol) const;

    /// <summary>
    /// Row-major order
    /// </summary>
    double &operator[](int aFlatIndex);

    /// <summary>
    /// Row-major order
    /// </summary>
    double const &operator[](int aFlatIndex) const;

    /// <summary>
    /// Row-major order
    /// </summary>
    double &operator[](size_t aFlatIndex);

    /// <summary>
    /// Row-major order
    /// </summary>
    double const &operator[](size_t aFlatIndex) const;

    /// <summary>
    /// Pointer to inner data array
    /// </summary>
    [[nodiscard]] const double *Data() const;

    /// <summary>
    /// Pointer to inner data array
    /// </summary>
    [[nodiscard]] double *Data();

    /// <summary>
    /// Element wise addition
    /// </summary>
    AffineTransformation &operator+=(double aOther);

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
    AffineTransformation &operator-=(double aOther);

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
    AffineTransformation &operator*=(double aOther);

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
    AffineTransformation &operator/=(double aOther);

    friend std::ostream &operator<<(std::ostream &aOs, const AffineTransformation &aMat);
    friend std::wostream &operator<<(std::wostream &aOs, const AffineTransformation &aMat);
    friend std::istream &operator>>(std::istream &aIs, AffineTransformation &aMat);
    friend std::wistream &operator>>(std::wistream &aIs, AffineTransformation &aMat);

    /// <summary>
    /// AffineTransformation inverse
    /// </summary>
    [[nodiscard]] AffineTransformation Inverse() const;

    /// <summary>
    /// AffineTransformation determinante
    /// </summary>
    [[nodiscard]] double Det() const;

    /// <summary>
    /// AffineTransformation trace (sum of the diagonal elements)
    /// </summary>
    [[nodiscard]] double Trace() const;

    /// <summary>
    /// The AffineTransformation diagonal elements
    /// </summary>
    [[nodiscard]] Vec3d Diagonal() const;

    /// <summary>
    /// AffineTransformation for image rotation
    /// </summary>
    [[nodiscard]] static AffineTransformation GetRotation(double aAngleInDeg) noexcept;
    /// <summary>
    /// AffineTransformation for image scaling
    /// </summary>
    [[nodiscard]] static AffineTransformation GetScale(double aScale) noexcept;
    /// <summary>
    /// AffineTransformation for image scaling (different for x and y)
    /// </summary>
    [[nodiscard]] static AffineTransformation GetScale(const Vec2d &aScale) noexcept;
    /// <summary>
    /// AffineTransformation for image translation
    /// </summary>
    [[nodiscard]] static AffineTransformation GetTranslation(const Vec2d &aShift) noexcept;
    /// <summary>
    /// AffineTransformation for image shearing
    /// </summary>
    [[nodiscard]] static AffineTransformation GetShear(const Vec2d &aShear) noexcept;
};

AffineTransformation operator+(const AffineTransformation &aLeft, double aRight);
AffineTransformation operator+(double aLeft, const AffineTransformation &aRight);
AffineTransformation operator-(const AffineTransformation &aLeft, double aRight);
AffineTransformation operator-(double aLeft, const AffineTransformation &aRight);

AffineTransformation operator*(const AffineTransformation &aLeft, double aRight);
AffineTransformation operator*(double aLeft, const AffineTransformation &aRight);
AffineTransformation operator/(const AffineTransformation &aLeft, double aRight);

/// <summary>
/// Matrix - vector multiplication <para/>
/// assuming vector is column vector
/// </summary>
Vec3d operator*(const AffineTransformation &aLeft, const Vec3d &aRight);

/// <summary>
/// Vector-Matrix multiplication <para/>
/// assuming vector is row vector
/// </summary>
Vec3d operator*(const Vec3d &aLeft, const AffineTransformation &aRight);

/// <summary>
/// affine matrix - vector multiplication <para/>
/// assuming vector is column vector and z element is 1<para/>
/// Implemented in header so that we have this operator also on GPU
/// </summary>
inline Vec2d operator*(const AffineTransformation &aLeft, const Vec2d &aRight)
{
    return Vec2d{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2],
                 aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5]};
}
} // namespace opp::image