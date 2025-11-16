#pragma once
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h>
#include <common/numeric_limits.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace mpp
{
template <AnyVector T> struct AbsDiff
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = T::AbsDiff(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.AbsDiff(aSrc1);
    }
};

template <AnyVector T> struct Add
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = aSrc1 + aSrc2;
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst += aSrc1;
    }
};

template <AnyVector T> struct Sub
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = aSrc1 - aSrc2;
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst -= aSrc1;
    }
};

/// <summary>
/// Inverted argument order for inplace sub: aSrcDst = aSrc1 - aSrcDst
/// </summary>
template <AnyVector T> struct SubInv
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.SubInv(aSrc1);
    }
};

template <AnyVector T> struct Mul
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = aSrc1 * aSrc2;
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst *= aSrc1;
    }
};

// Conjugate complex multiplication
template <ComplexVector T> struct ConjMul
{
    // aSrc1 * conj(aSrc2)
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = T::ConjMul(aSrc1, aSrc2);
    }
    // aSrcDst * conj(aSrc1)
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.ConjMul(aSrc1);
    }
};

// When result type is byte (unsigned char), then NPP handles seperatly the case that src1 == 0 and src2 == 0. Any
// number devided by 0 in floating point results in INF, but 0 / 0 gets NAN. Usually this would flush to 0 while
// casting to integer, but NPP returns 255...
template <AnyVector T, AnyVector DstT> struct Div
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = aSrc1 / aSrc2;
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst /= aSrc1;
    }
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires std::same_as<DstT, Vector1<byte>> || std::same_as<DstT, Vector3<byte>> ||
                 std::same_as<DstT, Vector4<byte>> // not Vector4A though
    {
        aDst = aSrc1;
        if (aDst.x == 0 && aSrc2.x == 0)
        {
            // by setting the first operand to 1 and keeping the second operand 0, the calculation will denormalize to
            // INF, any later scaling, if any, will keep the result as INF and the final result will always be 255, as
            // in NPP
            aDst.x = static_cast<remove_vector_t<T>>(1);
        }
        if constexpr (vector_active_size_v<T> > 1)
        {
            if (aDst.y == 0 && aSrc2.y == 0)
            {
                aDst.y = static_cast<remove_vector_t<T>>(1);
            }
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            if (aDst.z == 0 && aSrc2.z == 0)
            {
                aDst.z = static_cast<remove_vector_t<T>>(1);
            }
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            if (aDst.w == 0 && aSrc2.w == 0)
            {
                aDst.w = static_cast<remove_vector_t<T>>(1);
            }
        }
        aDst /= aSrc2;
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<DstT, Vector1<byte>> || std::same_as<DstT, Vector3<byte>> ||
                 std::same_as<DstT, Vector4<byte>> // not Vector4A though
    {
        if (aSrc1.x == 0 && aSrcDst.x == 0)
        {
            aSrcDst.x = static_cast<remove_vector_t<T>>(1);
        }
        if constexpr (vector_active_size_v<T> > 1)
        {
            if (aSrc1.y == 0 && aSrcDst.y == 0)
            {
                aSrcDst.y = static_cast<remove_vector_t<T>>(1);
            }
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            if (aSrc1.z == 0 && aSrcDst.z == 0)
            {
                aSrcDst.z = static_cast<remove_vector_t<T>>(1);
            }
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            if (aSrc1.w == 0 && aSrcDst.w == 0)
            {
                aSrcDst.w = static_cast<remove_vector_t<T>>(1);
            }
        }
        aSrcDst /= aSrc1;
    }
};

/// <summary>
/// Inverted argument order for inplace div: aSrcDst = aSrc1 / aSrcDst
/// </summary>
template <AnyVector T, AnyVector DstT> struct DivInv
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.DivInv(aSrc1);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
        requires std::same_as<DstT, Vector1<byte>> || std::same_as<DstT, Vector3<byte>> ||
                 std::same_as<DstT, Vector4<byte>> // not Vector4A though
    {
        T src1 = aSrc1;
        if (src1.x == 0 && aSrcDst.x == 0)
        {
            src1.x = static_cast<remove_vector_t<T>>(1);
        }
        if constexpr (vector_active_size_v<T> > 1)
        {
            if (src1.y == 0 && aSrcDst.y == 0)
            {
                src1.y = static_cast<remove_vector_t<T>>(1);
            }
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            if (src1.z == 0 && aSrcDst.z == 0)
            {
                src1.z = static_cast<remove_vector_t<T>>(1);
            }
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            if (src1.w == 0 && aSrcDst.w == 0)
            {
                src1.w = static_cast<remove_vector_t<T>>(1);
            }
        }
        aSrcDst.DivInv(src1);
    }
};

/// <summary>
/// Integer division with rounding
/// </summary>
template <RealOrComplexIntVector T, RoundingMode roundingMode> struct DivIntRound
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZero)
        {
            aDst = T::DivRound(aSrc1, aSrc2);
        }
        else if constexpr (roundingMode == RoundingMode::NearestTiesToEven)
        {
            aDst = T::DivRoundNearest(aSrc1, aSrc2);
        }
        else if constexpr (roundingMode == RoundingMode::TowardZero)
        {
            aDst = T::DivRoundZero(aSrc1, aSrc2);
        }
        else if constexpr (roundingMode == RoundingMode::TowardNegativeInfinity)
        {
            aDst = T::DivFloor(aSrc1, aSrc2);
        }
        else if constexpr (roundingMode == RoundingMode::TowardPositiveInfinity)
        {
            aDst = T::DivCeil(aSrc1, aSrc2);
        }
        else
        {
            static_assert(roundingMode != RoundingMode::NearestTiesAwayFromZero, "Unsupported rounding mode");
        }
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZero)
        {
            aSrcDst.DivRound(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::NearestTiesToEven)
        {
            aSrcDst.DivRoundNearest(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::TowardZero)
        {
            aSrcDst.DivRoundZero(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::TowardNegativeInfinity)
        {
            aSrcDst.DivFloor(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::TowardPositiveInfinity)
        {
            aSrcDst.DivCeil(aSrc1);
        }
        else
        {
            static_assert(roundingMode != RoundingMode::NearestTiesAwayFromZero, "Unsupported rounding mode");
        }
    }
};

/// <summary>
/// Integer division with rounding
/// Inverted argument order for inplace div: aSrcDst = aSrc1 / aSrcDst
/// </summary>
template <RealOrComplexIntVector T, RoundingMode roundingMode> struct DivInvIntRound
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZero)
        {
            aSrcDst.DivInvRound(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::NearestTiesToEven)
        {
            aSrcDst.DivInvRoundNearest(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::TowardZero)
        {
            aSrcDst.DivInvRoundZero(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::TowardNegativeInfinity)
        {
            aSrcDst.DivInvFloor(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::TowardPositiveInfinity)
        {
            aSrcDst.DivInvCeil(aSrc1);
        }
        else
        {
            static_assert(roundingMode != RoundingMode::NearestTiesAwayFromZero, "Unsupported rounding mode");
        }
    }
};

/// <summary>
/// Integer division with rounding. Multiplies by ScaleFactor before division.
/// </summary>
template <RealOrComplexIntVector T, RoundingMode roundingMode> struct DivIntScaleUpRound
{
    complex_basetype_t<image::pixel_basetype_t<T>> ScaleFactor;

    DivIntScaleUpRound(double aScaleFactor)
        : ScaleFactor(static_cast<complex_basetype_t<image::pixel_basetype_t<T>>>(aScaleFactor))
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZero)
        {
            aDst = aSrc1 * ScaleFactor;
            aDst.DivRound(aSrc2);
        }
        else if constexpr (roundingMode == RoundingMode::NearestTiesToEven)
        {
            aDst = aSrc1 * ScaleFactor;
            aDst.DivRoundNearest(aSrc2);
        }
        else if constexpr (roundingMode == RoundingMode::TowardZero)
        {
            aDst = aSrc1 * ScaleFactor;
            aDst.DivRoundZero(aSrc2);
        }
        else if constexpr (roundingMode == RoundingMode::TowardNegativeInfinity)
        {
            aDst = aSrc1 * ScaleFactor;
            aDst.DivFloor(aSrc2);
        }
        else if constexpr (roundingMode == RoundingMode::TowardPositiveInfinity)
        {
            aDst = aSrc1 * ScaleFactor;
            aDst.DivCeil(aSrc2);
        }
        else
        {
            static_assert(roundingMode != RoundingMode::NearestTiesAwayFromZero, "Unsupported rounding mode");
        }
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZero)
        {
            aSrcDst *= ScaleFactor;
            aSrcDst.DivRound(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::NearestTiesToEven)
        {
            aSrcDst *= ScaleFactor;
            aSrcDst.DivRoundNearest(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::TowardZero)
        {
            aSrcDst *= ScaleFactor;
            aSrcDst.DivRoundZero(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::TowardNegativeInfinity)
        {
            aSrcDst *= ScaleFactor;
            aSrcDst.DivFloor(aSrc1);
        }
        else if constexpr (roundingMode == RoundingMode::TowardPositiveInfinity)
        {
            aSrcDst *= ScaleFactor;
            aSrcDst.DivCeil(aSrc1);
        }
        else
        {
            static_assert(roundingMode != RoundingMode::NearestTiesAwayFromZero, "Unsupported rounding mode");
        }
    }
};

/// <summary>
/// Integer division with rounding. Multiplies by ScaleFactor before division.
/// Inverted argument order for inplace div: aSrcDst = aSrc1 / aSrcDst
/// </summary>
template <RealOrComplexIntVector T, RoundingMode roundingMode> struct DivInvIntScaleUpRound
{
    complex_basetype_t<image::pixel_basetype_t<T>> ScaleFactor;

    DivInvIntScaleUpRound(double aScaleFactor)
        : ScaleFactor(static_cast<complex_basetype_t<image::pixel_basetype_t<T>>>(aScaleFactor))
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZero)
        {
            aSrcDst.DivInvRound(aSrc1 * ScaleFactor);
        }
        else if constexpr (roundingMode == RoundingMode::NearestTiesToEven)
        {
            aSrcDst.DivInvRoundNearest(aSrc1 * ScaleFactor);
        }
        else if constexpr (roundingMode == RoundingMode::TowardZero)
        {
            aSrcDst.DivInvRoundZero(aSrc1 * ScaleFactor);
        }
        else if constexpr (roundingMode == RoundingMode::TowardNegativeInfinity)
        {
            aSrcDst.DivInvFloor(aSrc1 * ScaleFactor);
        }
        else if constexpr (roundingMode == RoundingMode::TowardPositiveInfinity)
        {
            aSrcDst.DivInvCeil(aSrc1 * ScaleFactor);
        }
        else
        {
            static_assert(roundingMode != RoundingMode::NearestTiesAwayFromZero, "Unsupported rounding mode");
        }
    }
};

template <RealIntVector T> struct And
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = T::And(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.And(aSrc1);
    }
};

template <RealIntVector T> struct Or
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = T::Or(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.Or(aSrc1);
    }
};

template <RealIntVector T> struct Xor
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = T::Xor(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.Xor(aSrc1);
    }
};

template <RealIntVector T> struct LShift
{
    const uint Shift;

    LShift(uint aShift) : Shift(aShift)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::LShift(aSrc1, Shift);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.LShift(Shift);
    }
};

template <RealIntVector T> struct RShift
{
    const uint Shift;

    RShift(uint aShift) : Shift(aShift)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::RShift(aSrc1, Shift);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.RShift(Shift);
    }
};

template <AnyVector T> struct AddSqr
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst += T::Sqr(aSrc1);
    }
};

template <RealVector T> struct Min
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = T::Min(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.Min(aSrc1);
    }
};

template <RealVector T> struct Max
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst = T::Max(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.Max(aSrc1);
    }
};

template <RealOrComplexFloatingVector T> struct EqEps
{
    complex_basetype_t<remove_vector_t<T>> eps;

    EqEps(complex_basetype_t<remove_vector_t<T>> aEps) : eps(aEps)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, Vector1<byte> &aDst) const
    {
        aDst = static_cast<byte>(static_cast<int>(T::EqEps(aSrc1, aSrc2, eps)) * TRUE_VALUE);
    }
};

template <AnyVector T> struct Eq
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, Vector1<byte> &aDst) const
    {
        aDst = static_cast<byte>(static_cast<int>(aSrc1 == aSrc2) * TRUE_VALUE);
    }
};

template <RealVector T> struct Ge
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, Vector1<byte> &aDst) const
    {
        aDst = static_cast<byte>(static_cast<int>(aSrc1 >= aSrc2) * TRUE_VALUE);
    }
};

template <RealVector T> struct Gt
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, Vector1<byte> &aDst) const
    {
        aDst = static_cast<byte>(static_cast<int>(aSrc1 > aSrc2) * TRUE_VALUE);
    }
};

template <RealVector T> struct Le
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, Vector1<byte> &aDst) const
    {
        aDst = static_cast<byte>(static_cast<int>(aSrc1 <= aSrc2) * TRUE_VALUE);
    }
};

template <RealVector T> struct Lt
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, Vector1<byte> &aDst) const
    {
        aDst = static_cast<byte>(static_cast<int>(aSrc1 < aSrc2) * TRUE_VALUE);
    }
};

template <AnyVector T> struct NEq
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, Vector1<byte> &aDst) const
    {
        aDst = static_cast<byte>(static_cast<int>(aSrc1 != aSrc2) * TRUE_VALUE);
    }
};

template <AnyVector T> struct CompareEq
{
    void operator()(const T &aSrc1, const T &aSrc2, same_vector_size_different_type_t<T, byte> &aDst) const
    {
        aDst = T::CompareEQ(aSrc1, aSrc2);
    }
};

template <RealVector T> struct CompareGe
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, same_vector_size_different_type_t<T, byte> &aDst) const
    {
        aDst = T::CompareGE(aSrc1, aSrc2);
    }
};

template <RealVector T> struct CompareGt
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, same_vector_size_different_type_t<T, byte> &aDst) const
    {
        aDst = T::CompareGT(aSrc1, aSrc2);
    }
};

template <RealVector T> struct CompareLe
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, same_vector_size_different_type_t<T, byte> &aDst) const
    {
        aDst = T::CompareLE(aSrc1, aSrc2);
    }
};

template <RealVector T> struct CompareLt
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, same_vector_size_different_type_t<T, byte> &aDst) const
    {
        aDst = T::CompareLT(aSrc1, aSrc2);
    }
};

template <AnyVector T> struct CompareNEq
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, same_vector_size_different_type_t<T, byte> &aDst) const
    {
        aDst = T::CompareNEQ(aSrc1, aSrc2);
    }
};

template <RealVector T, AlphaCompositionOp alphaOp> struct AlphaCompositionC
{
    const remove_vector_t<T> alphaSrc1; // Must be scaled to 0..1
    const remove_vector_t<T> alphaSrc2; // Must be scaled to 0..1

    AlphaCompositionC(remove_vector_t<T> aAlphaSrc1, remove_vector_t<T> aAlphaSrc2)
        : alphaSrc1(aAlphaSrc1), alphaSrc2(aAlphaSrc2)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
    {
        aDst.x = ComputePixel(aSrc1.x, aSrc2.x);
        if constexpr (vector_size_v<T> >= 2)
        {
            aDst.y = ComputePixel(aSrc1.y, aSrc2.y);
        }
        if constexpr (vector_size_v<T> >= 3)
        {
            aDst.z = ComputePixel(aSrc1.z, aSrc2.z);
        }
        if constexpr (image::FourChannelNoAlpha<T>)
        {
            aDst.w = ComputePixel(aSrc1.w, aSrc2.w);
        }
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.x = ComputePixel(aSrcDst.x, aSrc1.x);
        if constexpr (vector_size_v<T> >= 2)
        {
            aSrcDst.y = ComputePixel(aSrcDst.y, aSrc1.y);
        }
        if constexpr (vector_size_v<T> >= 3)
        {
            aSrcDst.z = ComputePixel(aSrcDst.z, aSrc1.z);
        }
        if constexpr (image::FourChannelNoAlpha<T>)
        {
            aSrcDst.w = ComputePixel(aSrcDst.w, aSrc1.w);
        }
    }

  private:
    DEVICE_CODE remove_vector_t<T> ComputePixel(remove_vector_t<T> aSrc1, remove_vector_t<T> aSrc2) const
    {
        switch (alphaOp)
        {
            case mpp::AlphaCompositionOp::Over:
                return alphaSrc1 * aSrc1 + (static_cast<remove_vector_t<T>>(1) - alphaSrc1) * alphaSrc2 * aSrc2;
            case mpp::AlphaCompositionOp::In:
                return alphaSrc1 * aSrc1 * alphaSrc2;
            case mpp::AlphaCompositionOp::Out:
                return alphaSrc1 * aSrc1 * (static_cast<remove_vector_t<T>>(1) - alphaSrc2);
            case mpp::AlphaCompositionOp::ATop:
                return alphaSrc1 * aSrc1 * alphaSrc2 +
                       (static_cast<remove_vector_t<T>>(1) - alphaSrc1) * alphaSrc2 * aSrc2;
            case mpp::AlphaCompositionOp::XOr:
                return alphaSrc1 * aSrc1 * (static_cast<remove_vector_t<T>>(1) - alphaSrc2) +
                       (static_cast<remove_vector_t<T>>(1) - alphaSrc1) * alphaSrc2 * aSrc2;
            case mpp::AlphaCompositionOp::Plus:
                return alphaSrc1 * aSrc1 + alphaSrc2 * aSrc2;
            case mpp::AlphaCompositionOp::OverPremul:
                return aSrc1 + (static_cast<remove_vector_t<T>>(1) - alphaSrc1) * aSrc2;
            case mpp::AlphaCompositionOp::InPremul:
                return aSrc1 * alphaSrc2;
            case mpp::AlphaCompositionOp::OutPremul:
                return aSrc1 * (static_cast<remove_vector_t<T>>(1) - alphaSrc2);
            case mpp::AlphaCompositionOp::ATopPremul:
                return aSrc1 * alphaSrc2 + (static_cast<remove_vector_t<T>>(1) - alphaSrc1) * aSrc2;
            case mpp::AlphaCompositionOp::XOrPremul:
                return aSrc1 * (static_cast<remove_vector_t<T>>(1) - alphaSrc2) +
                       (static_cast<remove_vector_t<T>>(1) - alphaSrc1) * aSrc2;
            case mpp::AlphaCompositionOp::PlusPremul:
                return aSrc1 + aSrc2;
        }
        return remove_vector_t<T>();
    }
};

template <RealVector T, AlphaCompositionOp alphaOp> struct AlphaComposition
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires(vector_size_v<T> == 1)
    {
        // One channel version, so only alpha? Doesn't exist in NPP but adding it makes AlphaComposition valid for all
        // vector sizes
        const remove_vector_t<T> alphaSrc1 = aSrc1.x;
        const remove_vector_t<T> alphaSrc2 = aSrc2.x;

        aDst.x = ComputeAlpha(alphaSrc1, alphaSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc2, T &aSrcDst) const
        requires(vector_size_v<T> == 1)
    {
        const remove_vector_t<T> alphaSrc1 = aSrcDst.x;
        const remove_vector_t<T> alphaSrc2 = aSrc2.x;

        aSrcDst.x = ComputeAlpha(alphaSrc1, alphaSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires(vector_size_v<T> == 2)
    {
        // Two channel version is named nppiAlphaComp_8u_AC1R, but it is a 2 channel function...
        const remove_vector_t<T> alphaSrc1 = aSrc1.y;
        const remove_vector_t<T> alphaSrc2 = aSrc2.y;

        aDst.x = ComputePixel(aSrc1.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aDst.y = ComputeAlpha(alphaSrc1, alphaSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc2, T &aSrcDst) const
        requires(vector_size_v<T> == 2)
    {
        const remove_vector_t<T> alphaSrc1 = aSrcDst.y;
        const remove_vector_t<T> alphaSrc2 = aSrc2.y;

        aSrcDst.x = ComputePixel(aSrcDst.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aSrcDst.y = ComputeAlpha(alphaSrc1, alphaSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires(vector_size_v<T> == 3)
    {
        // Three channel version doesn't exist in NPP, but for completness...
        const remove_vector_t<T> alphaSrc1 = aSrc1.z;
        const remove_vector_t<T> alphaSrc2 = aSrc2.z;

        aDst.x = ComputePixel(aSrc1.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aDst.y = ComputePixel(aSrc1.y, aSrc2.y, alphaSrc1, alphaSrc2);
        aDst.z = ComputeAlpha(alphaSrc1, alphaSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc2, T &aSrcDst) const
        requires(vector_size_v<T> == 3)
    {
        const remove_vector_t<T> alphaSrc1 = aSrcDst.z;
        const remove_vector_t<T> alphaSrc2 = aSrc2.z;

        aSrcDst.x = ComputePixel(aSrcDst.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aSrcDst.y = ComputePixel(aSrcDst.y, aSrc2.y, alphaSrc1, alphaSrc2);
        aSrcDst.z = ComputeAlpha(alphaSrc1, alphaSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires(vector_size_v<T> == 4)
    {
        const remove_vector_t<T> alphaSrc1 = aSrc1.w;
        const remove_vector_t<T> alphaSrc2 = aSrc2.w;

        aDst.x = ComputePixel(aSrc1.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aDst.y = ComputePixel(aSrc1.y, aSrc2.y, alphaSrc1, alphaSrc2);
        aDst.z = ComputePixel(aSrc1.z, aSrc2.z, alphaSrc1, alphaSrc2);
        aDst.w = ComputeAlpha(alphaSrc1, alphaSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc2, T &aSrcDst) const
        requires(vector_size_v<T> == 4)
    {
        const remove_vector_t<T> alphaSrc1 = aSrcDst.w;
        const remove_vector_t<T> alphaSrc2 = aSrc2.w;

        aSrcDst.x = ComputePixel(aSrcDst.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aSrcDst.y = ComputePixel(aSrcDst.y, aSrc2.y, alphaSrc1, alphaSrc2);
        aSrcDst.z = ComputePixel(aSrcDst.z, aSrc2.z, alphaSrc1, alphaSrc2);
        aSrcDst.w = ComputeAlpha(alphaSrc1, alphaSrc2);
    }

  protected:
    DEVICE_CODE remove_vector_t<T> ComputePixel(remove_vector_t<T> aSrc1, remove_vector_t<T> aSrc2,
                                                remove_vector_t<T> aAlpha1, remove_vector_t<T> aAlpha2) const
    {
        switch (alphaOp)
        {
            case mpp::AlphaCompositionOp::Over:
                return aAlpha1 * aSrc1 + (static_cast<remove_vector_t<T>>(1) - aAlpha1) * aAlpha2 * aSrc2;
            case mpp::AlphaCompositionOp::In:
                return aAlpha1 * aSrc1 * aAlpha2;
            case mpp::AlphaCompositionOp::Out:
                return aAlpha1 * aSrc1 * (static_cast<remove_vector_t<T>>(1) - aAlpha2);
            case mpp::AlphaCompositionOp::ATop:
                return aAlpha1 * aSrc1 * aAlpha2 + (static_cast<remove_vector_t<T>>(1) - aAlpha1) * aAlpha2 * aSrc2;
            case mpp::AlphaCompositionOp::XOr:
                return aAlpha1 * aSrc1 * (static_cast<remove_vector_t<T>>(1) - aAlpha2) +
                       (static_cast<remove_vector_t<T>>(1) - aAlpha1) * aAlpha2 * aSrc2;
            case mpp::AlphaCompositionOp::Plus:
                return aAlpha1 * aSrc1 + aAlpha2 * aSrc2;
            case mpp::AlphaCompositionOp::OverPremul:
                return aSrc1 + (static_cast<remove_vector_t<T>>(1) - aAlpha1) * aSrc2;
            case mpp::AlphaCompositionOp::InPremul:
                return aSrc1 * aAlpha2;
            case mpp::AlphaCompositionOp::OutPremul:
                return aSrc1 * (static_cast<remove_vector_t<T>>(1) - aAlpha2);
            case mpp::AlphaCompositionOp::ATopPremul:
                return aSrc1 * aAlpha2 + (static_cast<remove_vector_t<T>>(1) - aAlpha1) * aSrc2;
            case mpp::AlphaCompositionOp::XOrPremul:
                return aSrc1 * (static_cast<remove_vector_t<T>>(1) - aAlpha2) +
                       (static_cast<remove_vector_t<T>>(1) - aAlpha1) * aSrc2;
            case mpp::AlphaCompositionOp::PlusPremul:
                return aSrc1 + aSrc2;
        }
        return remove_vector_t<T>();
    }
    DEVICE_CODE remove_vector_t<T> ComputeAlpha(remove_vector_t<T> aAlpha1, remove_vector_t<T> aAlpha2) const
    {
        switch (alphaOp)
        {
            case mpp::AlphaCompositionOp::Over:
            case mpp::AlphaCompositionOp::OverPremul:
                return aAlpha1 + (static_cast<remove_vector_t<T>>(1) - aAlpha1) * aAlpha2;
            case mpp::AlphaCompositionOp::In:
            case mpp::AlphaCompositionOp::InPremul:
                return aAlpha1 * aAlpha2;
            case mpp::AlphaCompositionOp::Out:
            case mpp::AlphaCompositionOp::OutPremul:
                return aAlpha1 * (static_cast<remove_vector_t<T>>(1) - aAlpha2);
            case mpp::AlphaCompositionOp::ATop:
            case mpp::AlphaCompositionOp::ATopPremul:
                return aAlpha1 * aAlpha2 + (static_cast<remove_vector_t<T>>(1) - aAlpha1) * aAlpha2;
            case mpp::AlphaCompositionOp::XOr:
            case mpp::AlphaCompositionOp::XOrPremul:
                return aAlpha1 * (static_cast<remove_vector_t<T>>(1) - aAlpha2) +
                       (static_cast<remove_vector_t<T>>(1) - aAlpha1) * aAlpha2;
            case mpp::AlphaCompositionOp::Plus:
            case mpp::AlphaCompositionOp::PlusPremul:
            {
                if constexpr (Is16BitFloat<remove_vector_t<T>>)
                {
                    return remove_vector_t<T>::Min(aAlpha1 + aAlpha2, static_cast<remove_vector_t<T>>(1));
                }
                else
                {
#ifdef IS_HOST_COMPILER
                    return std::min(aAlpha1 + aAlpha2, static_cast<remove_vector_t<T>>(1));
#else
                    return min(aAlpha1 + aAlpha2, static_cast<remove_vector_t<T>>(1));
#endif
                }
            }
        }
        return remove_vector_t<T>();
    }
};

// For integer types with scaling alpha to float in value range 0..1
template <RealVector T, remove_vector_t<T> alphaMax, remove_vector_t<T> alphaMaxInv, AlphaCompositionOp alphaOp>
struct AlphaCompositionScale : public AlphaComposition<T, alphaOp>
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires(vector_size_v<T> == 1)
    {
        // One channel version, so only alpha? Doesn't exist in NPP but adding it makes AlphaComposition valid for all
        // vector sizes
        const remove_vector_t<T> alphaSrc1 = aSrc1.x * alphaMaxInv;
        const remove_vector_t<T> alphaSrc2 = aSrc2.x * alphaMaxInv;

        aDst.x = AlphaComposition<T, alphaOp>::ComputeAlpha(alphaSrc1, alphaSrc2) * alphaMax;
    }
    DEVICE_CODE void operator()(const T &aSrc2, T &aSrcDst) const
        requires(vector_size_v<T> == 1)
    {
        const remove_vector_t<T> alphaSrc1 = aSrcDst.x * alphaMaxInv;
        const remove_vector_t<T> alphaSrc2 = aSrc2.x * alphaMaxInv;

        aSrcDst.x = AlphaComposition<T, alphaOp>::ComputeAlpha(alphaSrc1, alphaSrc2) * alphaMax;
    }
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires(vector_size_v<T> == 2)
    {
        // Two channel version is named nppiAlphaComp_8u_AC1R, but it is a 2 channel function...
        const remove_vector_t<T> alphaSrc1 = aSrc1.y * alphaMaxInv;
        const remove_vector_t<T> alphaSrc2 = aSrc2.y * alphaMaxInv;

        aDst.x = AlphaComposition<T, alphaOp>::ComputePixel(aSrc1.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aDst.y = AlphaComposition<T, alphaOp>::ComputeAlpha(alphaSrc1, alphaSrc2) * alphaMax;
    }
    DEVICE_CODE void operator()(const T &aSrc2, T &aSrcDst) const
        requires(vector_size_v<T> == 2)
    {
        const remove_vector_t<T> alphaSrc1 = aSrcDst.y * alphaMaxInv;
        const remove_vector_t<T> alphaSrc2 = aSrc2.y * alphaMaxInv;

        aSrcDst.x = AlphaComposition<T, alphaOp>::ComputePixel(aSrcDst.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aSrcDst.y = AlphaComposition<T, alphaOp>::ComputeAlpha(alphaSrc1, alphaSrc2) * alphaMax;
    }
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires(vector_size_v<T> == 3)
    {
        // Three channel version doesn't exist in NPP, but for completness...
        const remove_vector_t<T> alphaSrc1 = aSrc1.z * alphaMaxInv;
        const remove_vector_t<T> alphaSrc2 = aSrc2.z * alphaMaxInv;

        aDst.x = AlphaComposition<T, alphaOp>::ComputePixel(aSrc1.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aDst.y = AlphaComposition<T, alphaOp>::ComputePixel(aSrc1.y, aSrc2.y, alphaSrc1, alphaSrc2);
        aDst.z = AlphaComposition<T, alphaOp>::ComputeAlpha(alphaSrc1, alphaSrc2) * alphaMax;
    }
    DEVICE_CODE void operator()(const T &aSrc2, T &aSrcDst) const
        requires(vector_size_v<T> == 3)
    {
        const remove_vector_t<T> alphaSrc1 = aSrcDst.z * alphaMaxInv;
        const remove_vector_t<T> alphaSrc2 = aSrc2.z * alphaMaxInv;

        aSrcDst.x = AlphaComposition<T, alphaOp>::ComputePixel(aSrcDst.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aSrcDst.y = AlphaComposition<T, alphaOp>::ComputePixel(aSrcDst.y, aSrc2.y, alphaSrc1, alphaSrc2);
        aSrcDst.z = AlphaComposition<T, alphaOp>::ComputeAlpha(alphaSrc1, alphaSrc2) * alphaMax;
    }
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst) const
        requires(vector_size_v<T> == 4)
    {
        const remove_vector_t<T> alphaSrc1 = aSrc1.w * alphaMaxInv;
        const remove_vector_t<T> alphaSrc2 = aSrc2.w * alphaMaxInv;

        aDst.x = AlphaComposition<T, alphaOp>::ComputePixel(aSrc1.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aDst.y = AlphaComposition<T, alphaOp>::ComputePixel(aSrc1.y, aSrc2.y, alphaSrc1, alphaSrc2);
        aDst.z = AlphaComposition<T, alphaOp>::ComputePixel(aSrc1.z, aSrc2.z, alphaSrc1, alphaSrc2);
        aDst.w = AlphaComposition<T, alphaOp>::ComputeAlpha(alphaSrc1, alphaSrc2) * alphaMax;
    }
    DEVICE_CODE void operator()(const T &aSrc2, T &aSrcDst) const
        requires(vector_size_v<T> == 4)
    {
        const remove_vector_t<T> alphaSrc1 = aSrcDst.w * alphaMaxInv;
        const remove_vector_t<T> alphaSrc2 = aSrc2.w * alphaMaxInv;

        aSrcDst.x = AlphaComposition<T, alphaOp>::ComputePixel(aSrcDst.x, aSrc2.x, alphaSrc1, alphaSrc2);
        aSrcDst.y = AlphaComposition<T, alphaOp>::ComputePixel(aSrcDst.y, aSrc2.y, alphaSrc1, alphaSrc2);
        aSrcDst.z = AlphaComposition<T, alphaOp>::ComputePixel(aSrcDst.z, aSrc2.z, alphaSrc1, alphaSrc2);
        aSrcDst.w = AlphaComposition<T, alphaOp>::ComputeAlpha(alphaSrc1, alphaSrc2) * alphaMax;
    }
};
} // namespace mpp
