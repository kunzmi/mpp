#include "affineTransformation.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/exception.h>
#include <common/image/solve.h>
#include <common/safeCast.h>
#include <common/utilities.h>
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
constexpr size_t AffineTransformation::GetIndex(int aRow, int aCol)
{
    assert(aRow >= 0);
    assert(aCol >= 0);
    assert(aRow < mRows);
    assert(aCol < mCols);

    return to_size_t(aCol + aRow * mCols);
}

AffineTransformation::AffineTransformation() noexcept : mData()
{
    mData[0] = 1.0;
    mData[4] = 1.0;
    mData[1] = 0.0;
    mData[2] = 0.0;
    mData[3] = 0.0;
    mData[5] = 0.0;
}

AffineTransformation::AffineTransformation(double aValues[mSize]) noexcept : mData()
{
    std::copy(aValues, aValues + mSize, mData);
}

AffineTransformation::AffineTransformation(double aX) noexcept : mData()
{
    mData[0] = aX;
    mData[1] = aX;
    mData[2] = aX;
    mData[3] = aX;
    mData[4] = aX;
    mData[5] = aX;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
AffineTransformation::AffineTransformation(double a00, double a01, double a02, double a10, double a11,
                                           double a12) noexcept
    : mData()
{
    mData[GetIndex(0, 0)] = a00; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(0, 1)] = a01; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(0, 2)] = a02; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(1, 0)] = a10; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(1, 1)] = a11; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(1, 2)] = a12; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
}

AffineTransformation::AffineTransformation(const std::pair<Vec2d, Vec2d> &aP0, const std::pair<Vec2d, Vec2d> &aP1,
                                           const std::pair<Vec2d, Vec2d> &aP2) noexcept
    : mData()
{
    bool ok = false;
    try
    {
        /*
         * Coefficients are calculated by solving linear system:
         * / x0 y0  1  0  0  0 \ /c00\ /u0\
         * | x1 y1  1  0  0  0 | |c01| |u1|
         * | x2 y2  1  0  0  0 | |c02| |u2|
         * |  0  0  0 x0 y0  1 | |c10| |v0|
         * |  0  0  0 x1 y1  1 | |c11| |v1|
         * \  0  0  0 x2 y2  1 / |c12| |v2|
         */
        std::vector<double> matrix = {
            aP0.first.x, aP0.first.y, 1, 0,           0,           0, aP0.second.x, //
            aP1.first.x, aP1.first.y, 1, 0,           0,           0, aP1.second.x, //
            aP2.first.x, aP2.first.y, 1, 0,           0,           0, aP2.second.x, //
            0,           0,           0, aP0.first.x, aP0.first.y, 1, aP0.second.y, //
            0,           0,           0, aP1.first.x, aP1.first.y, 1, aP1.second.y, //
            0,           0,           0, aP2.first.x, aP2.first.y, 1, aP2.second.y  //
        };

        ok = solve(matrix.data(), mData, 6);
    }
    catch (...)
    {
        ok = false;
    }

    if (!ok) // failed to solve system of equation
    {
        // set to unit transformation:
        mData[0] = 1.0;
        mData[4] = 1.0;
        mData[1] = 0.0;
        mData[2] = 0.0;
        mData[3] = 0.0;
        mData[5] = 0.0;
    }
}

const double *AffineTransformation::Data() const
{
    return mData;
}

double *AffineTransformation::Data()
{
    return mData;
}

bool AffineTransformation::operator==(const AffineTransformation &aOther) const
{
    bool ret = true;
    for (size_t i = 0; i < mSize; i++)
    {
        ret &= mData[i] == aOther[i]; // NOLINT --> non constant array index
    }
    return ret;
}
bool AffineTransformation::operator!=(const AffineTransformation &aOther) const
{
    return !(*this == aOther);
}

double &AffineTransformation::operator()(int aRow, int aCol)
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT --> non constant array index
}

double const &AffineTransformation::operator()(int aRow, int aCol) const
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT --> non constant array index
}

double &AffineTransformation::operator[](int aFlatIndex)
{
    assert(aFlatIndex >= 0);
    assert(aFlatIndex < mSize);
    return mData[to_size_t(aFlatIndex)]; // NOLINT --> non constant array index
}

double const &AffineTransformation::operator[](int aFlatIndex) const
{
    assert(aFlatIndex >= 0);
    assert(aFlatIndex < mSize);
    return mData[to_size_t(aFlatIndex)]; // NOLINT --> non constant array index
}

double &AffineTransformation::operator[](size_t aFlatIndex)
{
    assert(to_int(aFlatIndex) < mSize);
    return mData[aFlatIndex]; // NOLINT --> non constant array index
}

double const &AffineTransformation::operator[](size_t aFlatIndex) const
{
    assert(to_int(aFlatIndex) < mSize);
    return mData[aFlatIndex]; // NOLINT --> non constant array index
}

AffineTransformation &AffineTransformation::operator+=(double aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN += aOther; });
    return *this;
}

AffineTransformation &AffineTransformation::operator+=(const AffineTransformation &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::plus<>{});
    return *this;
}

AffineTransformation AffineTransformation::operator+(const AffineTransformation &aOther) const
{
    AffineTransformation ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::plus<>{});
    return ret;
}

AffineTransformation &AffineTransformation::operator-=(double aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN -= aOther; });
    return *this;
}

AffineTransformation &AffineTransformation::operator-=(const AffineTransformation &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::minus<>{});
    return *this;
}

AffineTransformation AffineTransformation::operator-(const AffineTransformation &aOther) const
{
    AffineTransformation ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::minus<>{});
    return ret;
}

AffineTransformation &AffineTransformation::operator*=(double aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN *= aOther; });
    return *this;
}

AffineTransformation &AffineTransformation::operator/=(double aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN /= aOther; });
    return *this;
}

AffineTransformation &AffineTransformation::operator*=(const AffineTransformation &aOther)
{
    const AffineTransformation ret = *this * aOther;
    *this                          = ret;
    return *this;
}

AffineTransformation AffineTransformation::operator*(const AffineTransformation &aOther) const
{
    AffineTransformation ret;

    ret[0] = mData[0] * aOther[0] + mData[1] * aOther[3];
    ret[1] = mData[0] * aOther[1] + mData[1] * aOther[4];
    ret[2] = mData[0] * aOther[2] + mData[1] * aOther[5] + mData[2];
    ret[3] = mData[3] * aOther[0] + mData[4] * aOther[3];
    ret[4] = mData[3] * aOther[1] + mData[4] * aOther[4];
    ret[5] = mData[3] * aOther[2] + mData[4] * aOther[5] + mData[5];

    return ret;
}

AffineTransformation AffineTransformation::Inverse() const
{
    const double det = Det();

    if (det == 0)
    {
        throw EXCEPTION("Cannot compute AffineTransformation inverse as determinant is 0.");
    }

    const double invdet = 1.0 / det;

    AffineTransformation ret;
    ret[0] = (mData[4]) * invdet;
    ret[1] = (-mData[1]) * invdet;
    ret[2] = (mData[1] * mData[5] - mData[2] * mData[4]) * invdet;
    ret[3] = (-mData[3]) * invdet;
    ret[4] = (mData[0]) * invdet;
    ret[5] = (mData[3] * mData[2] - mData[0] * mData[5]) * invdet;

    return ret;
}

double AffineTransformation::Det() const
{
    return mData[0] * (mData[4]) - mData[1] * (mData[3]);
}

double AffineTransformation::Trace() const
{
    return mData[0] + mData[4] + 1.0;
}

Vec3d AffineTransformation::Diagonal() const
{
    return {mData[0], mData[4], 1.0};
}

AffineTransformation AffineTransformation::GetRotation(double aAngleInDeg) noexcept
{
    AffineTransformation rot;
    const double angle = DEG_TO_RAD(aAngleInDeg);
    const double c     = std::cos(angle);
    const double s     = std::sin(angle);

    rot(0, 0) = c;
    rot(0, 1) = -s;
    rot(1, 0) = s;
    rot(1, 1) = c;

    return rot;
}

AffineTransformation AffineTransformation::GetTranslation(const Vec2d &aShift) noexcept
{
    AffineTransformation shift;
    shift(0, 2) = aShift.x;
    shift(1, 2) = aShift.y;

    return shift;
}

AffineTransformation AffineTransformation::GetScale(double aScale) noexcept
{
    AffineTransformation scale;
    scale(0, 0) = aScale;
    scale(1, 1) = aScale;

    return scale;
}

AffineTransformation AffineTransformation::GetScale(const Vec2d &aScale) noexcept
{
    AffineTransformation scale;
    scale(0, 0) = aScale.x;
    scale(1, 1) = aScale.y;

    return scale;
}

AffineTransformation AffineTransformation::GetShear(const Vec2d &aShear) noexcept
{
    AffineTransformation shear;
    shear(0, 1) = aShear.x;
    shear(1, 0) = aShear.y;

    return shear;
}

std::ostream &operator<<(std::ostream &aOs, const AffineTransformation &aMat)
{
    std::streamsize maxSize = 0;
    for (const auto &elem : aMat.mData)
    {
        maxSize = std::max(std::streamsize(std::to_string(elem).length()), maxSize);
    }

    // clang tidy gets a bit crazy with std::streamsize...
    // NOLINTBEGIN
    aOs << '(' << std::setw(maxSize) << std::to_string(aMat[0]) << " " << std::setw(maxSize) << std::to_string(aMat[1])
        << " " << std::setw(maxSize) << std::to_string(aMat[2]) << ')' << std::endl;
    aOs << '(' << std::setw(maxSize) << std::to_string(aMat[3]) << " " << std::setw(maxSize) << std::to_string(aMat[4])
        << " " << std::setw(maxSize) << std::to_string(aMat[5]) << ')' << std::endl;
    aOs << '(' << std::setw(maxSize) << std::to_string(0.0) << " " << std::setw(maxSize) << std::to_string(0.0) << " "
        << std::setw(maxSize) << std::to_string(1.0) << ')' << std::endl;
    return aOs;
    // NOLINTEND
}

std::wostream &operator<<(std::wostream &aOs, const AffineTransformation &aMat)
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
    aOs << '(' << std::setw(maxSize) << std::to_wstring(0.0) << " " << std::setw(maxSize) << std::to_wstring(0.0) << " "
        << std::setw(maxSize) << std::to_wstring(1.0) << ')' << std::endl;
    return aOs;
    // NOLINTEND
}

std::istream &operator>>(std::istream &aIs, AffineTransformation &aMat)
{
    for (size_t i = 0; i < to_size_t(AffineTransformation::mSize); i++)
    {
        aIs >> aMat[i];
    }
    return aIs;
}

std::wistream &operator>>(std::wistream &aIs, AffineTransformation &aMat)
{
    for (size_t i = 0; i < to_size_t(AffineTransformation::mSize); i++)
    {
        aIs >> aMat[i];
    }
    return aIs;
}

AffineTransformation operator+(const AffineTransformation &aLeft, double aRight)
{
    AffineTransformation ret(aLeft);
    ret += aRight;
    return ret;
}

AffineTransformation operator+(double aLeft, const AffineTransformation &aRight)
{
    AffineTransformation ret(aLeft);
    ret += aRight;
    return ret;
}

AffineTransformation operator-(const AffineTransformation &aLeft, double aRight)
{
    AffineTransformation ret(aLeft);
    ret -= aRight;
    return ret;
}

AffineTransformation operator-(double aLeft, const AffineTransformation &aRight)
{
    AffineTransformation ret(aLeft);
    ret -= aRight;
    return ret;
}

AffineTransformation operator*(const AffineTransformation &aLeft, double aRight)
{
    AffineTransformation ret(aLeft);
    ret *= aRight;
    return ret;
}

AffineTransformation operator*(double aLeft, const AffineTransformation &aRight)
{
    AffineTransformation ret(aRight);
    ret *= aLeft;
    return ret;
}

AffineTransformation operator/(const AffineTransformation &aLeft, double aRight)
{
    AffineTransformation ret(aLeft);
    ret *= 1.0 / aRight;
    return ret;
}

Vec3d operator*(const AffineTransformation &aLeft, const Vec3d &aRight)
{
    return Vec3d{aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z,
                 aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5] * aRight.z, aRight.z};
}

Vec3d operator*(const Vec3d &aLeft, const AffineTransformation &aRight)
{
    return Vec3d{aLeft.x * aRight[0] + aLeft.y * aRight[3], aLeft.x * aRight[1] + aLeft.y * aRight[4],
                 aLeft.x * aRight[2] + aLeft.y * aRight[5] + aLeft.z};
}

} // namespace opp::image