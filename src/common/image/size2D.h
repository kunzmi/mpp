#pragma once
#include <common/defines.h>
#include <common/vector2.h>
#include <istream>
#include <ostream>

namespace opp::image
{
/// <summary>
/// A specialized type to describe image size in 2D (number of pixels in X and Y)
/// Note: in device code, only the vector2<int> device code is available.
/// </summary>
class Size2D : public Vector2<int>
{
  public:
    /// <summary>
    /// Initializes a new size to all components = 0
    /// </summary>
    Size2D() noexcept;

    /// <summary>
    /// Initializes a new size to (aX, aY)
    /// </summary>
    Size2D(int aX, int aY) noexcept;

    /// <summary>
    /// Initializes a new size to aVec
    /// </summary>
    Size2D(const Vector2<int> &aVec) noexcept; // NOLINT(hicpp-explicit-conversions)

    /// <summary>
    /// Initializes a new size to aVec
    /// </summary>
    Size2D(int aArr[2]) noexcept; // NOLINT(hicpp-explicit-conversions)

    ~Size2D() = default;

    Size2D(const Size2D &)                     = default;
    Size2D(Size2D &&)                          = default;
    Size2D &operator=(const Size2D &) noexcept = default;
    Size2D &operator=(Size2D &&) noexcept      = default;

    Size2D &operator=(const Vector2<int> &aOther) noexcept;

    /// <summary>
    /// Component wise addition
    /// </summary>
    Size2D &operator+=(int aOther);

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    Size2D &operator-=(int aOther);

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    Size2D &operator*=(int aOther);

    /// <summary>
    /// Component wise division
    /// </summary>
    Size2D &operator/=(int aOther);

    /// <summary>
    /// Component wise addition
    /// </summary>
    Size2D &operator+=(const Vector2<int> &aOther);

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    Size2D &operator-=(const Vector2<int> &aOther);

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    Size2D &operator*=(const Vector2<int> &aOther);

    /// <summary>
    /// Component wise division
    /// </summary>
    Size2D &operator/=(const Vector2<int> &aOther);

    /// <summary>
    /// Converts a 2D pixel index (x, y) to a flattened array index
    /// </summary>
    [[nodiscard]] size_t operator()(const Vector2<int> &aCoords) const;

    /// <summary>
    /// Converts a 2D pixel index (x, y) to a flattened array index
    /// </summary>
    [[nodiscard]] size_t operator()(int aX, int aY) const;

    /// <summary>
    /// Converts a 2D pixel index (x, y) to a flattened array index
    /// </summary>
    [[nodiscard]] size_t operator()(size_t aX, size_t aY) const;

    /// <summary>
    /// Returns the total size (number of pixels)
    /// </summary>
    [[nodiscard]] size_t TotalSize() const;

    /// <summary>
    /// Returns the 2D coordinates of a flat index
    /// </summary>
    [[nodiscard]] Vector2<int> GetCoordinates(size_t aIndex) const;

    /// <summary>
    /// Returns the 2D coordinates of a flat index
    /// </summary>
    [[nodiscard]] Vector2<int> GetCoordinates(int aIndex) const;

    /// <summary>
    /// Converts a 2D pixel index (x, y) to a flattened array index
    /// </summary>
    [[nodiscard]] size_t GetFlatIndex(const Vector2<int> &aCoords) const;

    /// <summary>
    /// Converts a 2D pixel index (x, y) to a flattened array index
    /// </summary>
    [[nodiscard]] size_t GetFlatIndex(int aX, int aY) const;

    /// <summary>
    /// Converts a 2D pixel index (x, y) to a flattened array index
    /// </summary>
    [[nodiscard]] size_t GetFlatIndex(size_t aX, size_t aY) const;
};

Size2D operator+(const Size2D &aLeft, const Vector2<int> &aRight);
Size2D operator+(const Size2D &aLeft, const Size2D &aRight);
Size2D operator+(const Size2D &aLeft, int aRight);
Size2D operator+(int aLeft, const Size2D &aRight);
Size2D operator-(const Size2D &aLeft, const Vector2<int> &aRight);
Size2D operator-(const Size2D &aLeft, const Size2D &aRight);
Size2D operator-(const Size2D &aLeft, int aRight);
Size2D operator-(int aLeft, const Size2D &aRight);

Size2D operator*(const Size2D &aLeft, const Vector2<int> &aRight);
Size2D operator*(const Size2D &aLeft, const Size2D &aRight);
Size2D operator*(const Size2D &aLeft, int aRight);
Size2D operator*(int aLeft, const Size2D &aRight);
Size2D operator/(const Size2D &aLeft, const Vector2<int> &aRight);
Size2D operator/(const Size2D &aLeft, const Size2D &aRight);
Size2D operator/(const Size2D &aLeft, int aRight);
Size2D operator/(int aLeft, const Size2D &aRight);

std::ostream &operator<<(std::ostream &aOs, const Size2D &aSize);
std::wostream &operator<<(std::wostream &aOs, const Size2D &aSize);
std::istream &operator>>(std::istream &aIs, Size2D &aSize);
std::wistream &operator>>(std::wistream &aIs, Size2D &aSize);
} // namespace opp::image