#include "../dllexport_common.h"
#include "affineTransformation.h"
#include "quad.h"
#include "roi.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/exception.h>
#include <common/image/solve.h>
#include <common/numberTypes.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector2.h>
#include <common/vector3.h>
#include <concepts>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace mpp::image
{
template <RealFloatingPoint T> constexpr size_t AffineTransformation<T>::GetIndex(int aRow, int aCol)
{
    assert(aRow >= 0);
    assert(aCol >= 0);
    assert(aRow < mRows);
    assert(aCol < mCols);

    return to_size_t(aCol + aRow * mCols);
}

template <RealFloatingPoint T> AffineTransformation<T>::AffineTransformation() noexcept : mData()
{
    mData[0] = static_cast<T>(1.0);
    mData[4] = static_cast<T>(1.0);
    mData[1] = 0;
    mData[2] = 0;
    mData[3] = 0;
    mData[5] = 0;
}

template <RealFloatingPoint T> AffineTransformation<T>::AffineTransformation(T aValues[mSize]) noexcept : mData()
{
    std::copy(aValues, aValues + mSize, mData);
}

template <RealFloatingPoint T> AffineTransformation<T>::AffineTransformation(T aX) noexcept : mData()
{
    mData[0] = aX;
    mData[1] = aX;
    mData[2] = aX;
    mData[3] = aX;
    mData[4] = aX;
    mData[5] = aX;
}

template <RealFloatingPoint T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
AffineTransformation<T>::AffineTransformation(T a00, T a01, T a02, T a10, T a11, T a12) noexcept : mData()
{
    mData[GetIndex(0, 0)] = a00; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(0, 1)] = a01; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(0, 2)] = a02; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(1, 0)] = a10; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(1, 1)] = a11; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
    mData[GetIndex(1, 2)] = a12; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
}

template <RealFloatingPoint T>
AffineTransformation<T>::AffineTransformation(const std::pair<Vector2<T>, Vector2<T>> &aP0,
                                              const std::pair<Vector2<T>, Vector2<T>> &aP1,
                                              const std::pair<Vector2<T>, Vector2<T>> &aP2) noexcept
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
        std::vector<T> matrix = {
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
        mData[0] = 1;
        mData[4] = 1;
        mData[1] = 0;
        mData[2] = 0;
        mData[3] = 0;
        mData[5] = 0;
    }
}

template <RealFloatingPoint T>
AffineTransformation<T>::AffineTransformation(const Roi &aRoi, const Quad<T> &aQuad) noexcept
    : AffineTransformation(Quad<T>(aRoi), aQuad)
{
}

template <RealFloatingPoint T>
AffineTransformation<T>::AffineTransformation(const Quad<T> &aSrcQuad, const Quad<T> &aDstQuad) noexcept : mData()
{
    bool ok = false;
    try
    {
        const Quad<T> &src = aSrcQuad;
        const Quad<T> &dst = aDstQuad;

        /*
         * Coefficients are calculated by solving linear system:
         * / x0 y0  1  0  0  0 \ /c00\ /u0\
         * | x1 y1  1  0  0  0 | |c01| |u1|
         * | x2 y2  1  0  0  0 | |c02| |u2|
         * |  0  0  0 x0 y0  1 | |c10| |v0|
         * |  0  0  0 x1 y1  1 | |c11| |v1|
         * \  0  0  0 x2 y2  1 / |c12| |v2|
         */
        std::vector<T> matrix = {src.P0.x, src.P0.y, 1, 0,        0,        0, dst.P0.x, //
                                 src.P1.x, src.P1.y, 1, 0,        0,        0, dst.P1.x, //
                                 src.P2.x, src.P2.y, 1, 0,        0,        0, dst.P2.x, //
                                 0,        0,        0, src.P0.x, src.P0.y, 1, dst.P0.y, //
                                 0,        0,        0, src.P1.x, src.P1.y, 1, dst.P1.y, //
                                 0,        0,        0, src.P2.x, src.P2.y, 1, dst.P2.y};

        ok = solve(matrix.data(), mData, 6);
    }
    catch (...)
    {
        ok = false;
    }

    if (!ok) // failed to solve system of equation
    {
        // set to unit transformation:
        mData[0] = 1;
        mData[4] = 1;
        mData[1] = 0;
        mData[2] = 0;
        mData[3] = 0;
        mData[5] = 0;
    }
}

template <RealFloatingPoint T>
template <RealFloatingPoint T2>
AffineTransformation<T>::AffineTransformation(const AffineTransformation<T2> &aOther) noexcept
    requires(!std::same_as<T, T2>)
    : mData()
{
    mData[0] = static_cast<T>(aOther.Data()[0]);
    mData[1] = static_cast<T>(aOther.Data()[1]);
    mData[2] = static_cast<T>(aOther.Data()[2]);
    mData[3] = static_cast<T>(aOther.Data()[3]);
    mData[4] = static_cast<T>(aOther.Data()[4]);
    mData[5] = static_cast<T>(aOther.Data()[5]);
}

template <RealFloatingPoint T> const T *AffineTransformation<T>::Data() const
{
    return mData;
}

template <RealFloatingPoint T> T *AffineTransformation<T>::Data()
{
    return mData;
}

template <RealFloatingPoint T> bool AffineTransformation<T>::operator==(const AffineTransformation<T> &aOther) const
{
    bool ret = true;
    for (size_t i = 0; i < mSize; i++)
    {
        ret &= mData[i] == aOther[i]; // NOLINT --> non constant array index
    }
    return ret;
}
template <RealFloatingPoint T> bool AffineTransformation<T>::operator!=(const AffineTransformation<T> &aOther) const
{
    return !(*this == aOther);
}

template <RealFloatingPoint T> T &AffineTransformation<T>::operator()(int aRow, int aCol)
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT --> non constant array index
}

template <RealFloatingPoint T> T const &AffineTransformation<T>::operator()(int aRow, int aCol) const
{
    return mData[GetIndex(aRow, aCol)]; // NOLINT --> non constant array index
}

template <RealFloatingPoint T> T &AffineTransformation<T>::operator[](int aFlatIndex)
{
    assert(aFlatIndex >= 0);
    assert(aFlatIndex < mSize);
    return mData[to_size_t(aFlatIndex)]; // NOLINT --> non constant array index
}

// template <RealFloatingPoint T> T const &AffineTransformation<T>::operator[](int aFlatIndex) const
//{
//     assert(aFlatIndex >= 0);
//     assert(aFlatIndex < mSize);
//     return mData[to_size_t(aFlatIndex)]; // NOLINT --> non constant array index
// }

template <RealFloatingPoint T> T &AffineTransformation<T>::operator[](size_t aFlatIndex)
{
    assert(to_int(aFlatIndex) < mSize);
    return mData[aFlatIndex]; // NOLINT --> non constant array index
}

// template <RealFloatingPoint T> T const &AffineTransformation<T>::operator[](size_t aFlatIndex) const
//{
//     assert(to_int(aFlatIndex) < mSize);
//     return mData[aFlatIndex]; // NOLINT --> non constant array index
// }

template <RealFloatingPoint T> AffineTransformation<T> &AffineTransformation<T>::operator+=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN += aOther; });
    return *this;
}

template <RealFloatingPoint T>
AffineTransformation<T> &AffineTransformation<T>::operator+=(const AffineTransformation<T> &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::plus<>{});
    return *this;
}

template <RealFloatingPoint T>
AffineTransformation<T> AffineTransformation<T>::operator+(const AffineTransformation<T> &aOther) const
{
    AffineTransformation ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::plus<>{});
    return ret;
}

template <RealFloatingPoint T> AffineTransformation<T> &AffineTransformation<T>::operator-=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN -= aOther; });
    return *this;
}

template <RealFloatingPoint T>
AffineTransformation<T> &AffineTransformation<T>::operator-=(const AffineTransformation<T> &aOther)
{
    std::transform(mData, mData + mSize, aOther.mData, mData, std::minus<>{});
    return *this;
}

template <RealFloatingPoint T>
AffineTransformation<T> AffineTransformation<T>::operator-(const AffineTransformation<T> &aOther) const
{
    AffineTransformation ret;
    std::transform(mData, mData + mSize, aOther.mData, ret.mData, std::minus<>{});
    return ret;
}

template <RealFloatingPoint T> AffineTransformation<T> &AffineTransformation<T>::operator*=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN *= aOther; });
    return *this;
}

template <RealFloatingPoint T> AffineTransformation<T> &AffineTransformation<T>::operator/=(T aOther)
{
    std::for_each(mData, mData + mSize, [&](auto &aN) { aN /= aOther; });
    return *this;
}

template <RealFloatingPoint T>
AffineTransformation<T> &AffineTransformation<T>::operator*=(const AffineTransformation<T> &aOther)
{
    const AffineTransformation ret = *this * aOther;
    *this                          = ret;
    return *this;
}

template <RealFloatingPoint T>
AffineTransformation<T> AffineTransformation<T>::operator*(const AffineTransformation<T> &aOther) const
{
    AffineTransformation<T> ret;

    ret[0] = mData[0] * aOther[0] + mData[1] * aOther[3];
    ret[1] = mData[0] * aOther[1] + mData[1] * aOther[4];
    ret[2] = mData[0] * aOther[2] + mData[1] * aOther[5] + mData[2];
    ret[3] = mData[3] * aOther[0] + mData[4] * aOther[3];
    ret[4] = mData[3] * aOther[1] + mData[4] * aOther[4];
    ret[5] = mData[3] * aOther[2] + mData[4] * aOther[5] + mData[5];

    return ret;
}

template <RealFloatingPoint T> AffineTransformation<T> AffineTransformation<T>::Inverse() const
{
    const T det = Det();

    if (det == 0)
    {
        throw EXCEPTION("Cannot compute AffineTransformation inverse as determinant is 0.");
    }

    const T invdet = static_cast<T>(1.0) / det;

    AffineTransformation<T> ret;
    ret[0] = (mData[4]) * invdet;
    ret[1] = (-mData[1]) * invdet;
    ret[2] = (mData[1] * mData[5] - mData[2] * mData[4]) * invdet;
    ret[3] = (-mData[3]) * invdet;
    ret[4] = (mData[0]) * invdet;
    ret[5] = (mData[3] * mData[2] - mData[0] * mData[5]) * invdet;

    return ret;
}

template <RealFloatingPoint T> T AffineTransformation<T>::Det() const
{
    return mData[0] * (mData[4]) - mData[1] * (mData[3]);
}

template <RealFloatingPoint T> T AffineTransformation<T>::Trace() const
{
    return mData[0] + mData[4] + static_cast<T>(1.0);
}

template <RealFloatingPoint T> Vector3<T> AffineTransformation<T>::Diagonal() const
{
    return {mData[0], mData[4], static_cast<T>(1.0)};
}

template <RealFloatingPoint T> AffineTransformation<T> AffineTransformation<T>::GetRotation(T aAngleInDeg) noexcept
{
    AffineTransformation<T> rot;
    const T angle = DEG_TO_RAD(aAngleInDeg);
    const T c     = std::cos(angle);
    const T s     = std::sin(angle);

    rot(0, 0) = c;
    rot(0, 1) = s;
    rot(1, 0) = -s;
    rot(1, 1) = c;

    return rot;
}

template <RealFloatingPoint T>
AffineTransformation<T> AffineTransformation<T>::GetTranslation(const Vector2<T> &aShift) noexcept
{
    AffineTransformation<T> shift;
    shift(0, 2) = aShift.x;
    shift(1, 2) = aShift.y;

    return shift;
}

template <RealFloatingPoint T> AffineTransformation<T> AffineTransformation<T>::GetScale(T aScale) noexcept
{
    AffineTransformation<T> scale;
    scale(0, 0) = aScale;
    scale(1, 1) = aScale;

    return scale;
}

template <RealFloatingPoint T>
AffineTransformation<T> AffineTransformation<T>::GetScale(const Vector2<T> &aScale) noexcept
{
    AffineTransformation<T> scale;
    scale(0, 0) = aScale.x;
    scale(1, 1) = aScale.y;

    return scale;
}

template <RealFloatingPoint T>
AffineTransformation<T> AffineTransformation<T>::GetShear(const Vector2<T> &aShear) noexcept
{
    AffineTransformation<T> shear;
    shear(0, 1) = aShear.x;
    shear(1, 0) = aShear.y;

    return shear;
}

template <RealFloatingPoint T> AffineTransformation<T> operator+(const AffineTransformation<T> &aLeft, T aRight)
{
    AffineTransformation<T> ret(aLeft);
    ret += aRight;
    return ret;
}

template <RealFloatingPoint T> AffineTransformation<T> operator+(T aLeft, const AffineTransformation<T> &aRight)
{
    AffineTransformation<T> ret(aLeft);
    ret += aRight;
    return ret;
}

template <RealFloatingPoint T> AffineTransformation<T> operator-(const AffineTransformation<T> &aLeft, T aRight)
{
    AffineTransformation<T> ret(aLeft);
    ret -= aRight;
    return ret;
}

template <RealFloatingPoint T> AffineTransformation<T> operator-(T aLeft, const AffineTransformation<T> &aRight)
{
    AffineTransformation<T> ret(aLeft);
    ret -= aRight;
    return ret;
}

template <RealFloatingPoint T> AffineTransformation<T> operator*(const AffineTransformation<T> &aLeft, T aRight)
{
    AffineTransformation<T> ret(aLeft);
    ret *= aRight;
    return ret;
}

template <RealFloatingPoint T> AffineTransformation<T> operator*(T aLeft, const AffineTransformation<T> &aRight)
{
    AffineTransformation<T> ret(aRight);
    ret *= aLeft;
    return ret;
}

template <RealFloatingPoint T> AffineTransformation<T> operator/(const AffineTransformation<T> &aLeft, T aRight)
{
    AffineTransformation<T> ret(aLeft);
    ret *= static_cast<T>(1.0) / aRight;
    return ret;
}

template <RealFloatingPoint T> Vector3<T> operator*(const AffineTransformation<T> &aLeft, const Vector3<T> &aRight)
{
    return {aLeft[0] * aRight.x + aLeft[1] * aRight.y + aLeft[2] * aRight.z,
            aLeft[3] * aRight.x + aLeft[4] * aRight.y + aLeft[5] * aRight.z, aRight.z};
}

template <RealFloatingPoint T> Vector3<T> operator*(const Vector3<T> &aLeft, const AffineTransformation<T> &aRight)
{
    return {aLeft.x * aRight[0] + aLeft.y * aRight[3], aLeft.x * aRight[1] + aLeft.y * aRight[4],
            aLeft.x * aRight[2] + aLeft.y * aRight[5] + aLeft.z};
}

template <RealFloatingPoint T> Quad<T> operator*(const AffineTransformation<T> &aLeft, const Quad<T> &aRight)
{
    return {aLeft * aRight.P0, aLeft * aRight.P1, aLeft * aRight.P2, aLeft * aRight.P3};
}

template <RealFloatingPoint T> Quad<T> operator*(const AffineTransformation<T> &aLeft, const Roi &aRight)
{
    return aLeft * Quad<T>(aRight);
}

// instantiate for float and double:
template class AffineTransformation<float>;
template class AffineTransformation<double>;

template MPPEXPORT_COMMON AffineTransformation<float>::AffineTransformation(const AffineTransformation<double> &aOther);
template MPPEXPORT_COMMON AffineTransformation<double>::AffineTransformation(const AffineTransformation<float> &aOther);

template AffineTransformation<float> MPPEXPORT_COMMON operator+(const AffineTransformation<float> &aLeft, float aRight);
template AffineTransformation<float> MPPEXPORT_COMMON operator+(float aLeft, const AffineTransformation<float> &aRight);
template AffineTransformation<float> MPPEXPORT_COMMON operator-(const AffineTransformation<float> &aLeft, float aRight);
template AffineTransformation<float> MPPEXPORT_COMMON operator-(float aLeft, const AffineTransformation<float> &aRight);

template AffineTransformation<float> MPPEXPORT_COMMON operator*(const AffineTransformation<float> &aLeft, float aRight);
template AffineTransformation<float> MPPEXPORT_COMMON operator*(float aLeft, const AffineTransformation<float> &aRight);
template AffineTransformation<float> MPPEXPORT_COMMON operator/(const AffineTransformation<float> &aLeft, float aRight);

template Vector3<float> MPPEXPORT_COMMON operator*(const AffineTransformation<float> &aLeft,
                                                   const Vector3<float> &aRight);
template Vector3<float> MPPEXPORT_COMMON operator*(const Vector3<float> &aLeft,
                                                   const AffineTransformation<float> &aRight);
template Quad<float> MPPEXPORT_COMMON operator*(const AffineTransformation<float> &aLeft, const Quad<float> &aRight);
template Quad<float> MPPEXPORT_COMMON operator*(const AffineTransformation<float> &aLeft, const Roi &aRight);

template AffineTransformation<double> MPPEXPORT_COMMON operator+(const AffineTransformation<double> &aLeft,
                                                                 double aRight);
template AffineTransformation<double> MPPEXPORT_COMMON operator+(double aLeft,
                                                                 const AffineTransformation<double> &aRight);
template AffineTransformation<double> MPPEXPORT_COMMON operator-(const AffineTransformation<double> &aLeft,
                                                                 double aRight);
template AffineTransformation<double> MPPEXPORT_COMMON operator-(double aLeft,
                                                                 const AffineTransformation<double> &aRight);

template AffineTransformation<double> MPPEXPORT_COMMON operator*(const AffineTransformation<double> &aLeft,
                                                                 double aRight);
template AffineTransformation<double> MPPEXPORT_COMMON operator*(double aLeft,
                                                                 const AffineTransformation<double> &aRight);
template AffineTransformation<double> MPPEXPORT_COMMON operator/(const AffineTransformation<double> &aLeft,
                                                                 double aRight);

template Vector3<double> MPPEXPORT_COMMON operator*(const AffineTransformation<double> &aLeft,
                                                    const Vector3<double> &aRight);
template Vector3<double> MPPEXPORT_COMMON operator*(const Vector3<double> &aLeft,
                                                    const AffineTransformation<double> &aRight);
template Quad<double> MPPEXPORT_COMMON operator*(const AffineTransformation<double> &aLeft, const Quad<double> &aRight);
template Quad<double> MPPEXPORT_COMMON operator*(const AffineTransformation<double> &aLeft, const Roi &aRight);

} // namespace mpp::image