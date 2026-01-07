#pragma once
#include "colorMatrices.h"
#include <common/defines.h>
#include <common/image/channel.h>
#include <common/image/matrix.h>
#include <common/image/matrix3x4.h>
#include <common/image/matrix4x4.h>
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace mpp::image
{

template <RealVector T, typename TOp> struct SetAlphaConst
{
    remove_vector_t<T> Alpha;
    TOp op;

    SetAlphaConst(remove_vector_t<T> aAlpha, TOp aOp) : Alpha(aAlpha), op(aOp)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        const Vector4A<remove_vector_t<T>> src(aSrc1);
        Vector4A<remove_vector_t<T>> dst;
        op(src, dst);
        aDst.x = dst.x;
        aDst.y = dst.y;
        aDst.z = dst.z;
        aDst.w = Alpha;
    }

    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        op(aSrcDst);
        aSrcDst.w = Alpha;
    }
};

template <RealVector T, typename TOp> struct SetAlpha
{
    TOp op;

    SetAlpha(TOp aOp) : op(aOp)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        const Vector4A<remove_vector_t<T>> src(aSrc1);
        Vector4A<remove_vector_t<T>> dst;
        op(src, dst);
        aDst.x = dst.x;
        aDst.y = dst.y;
        aDst.z = dst.z;
        aDst.w = aSrc1.w;
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        // in case that op does not preserve alpha channel:
        const remove_vector_t<T> alpha = aSrcDst.w;

        Vector4A<remove_vector_t<T>> src(aSrcDst);
        op(src);
        aSrcDst.x = src.x;
        aSrcDst.y = src.y;
        aSrcDst.z = src.z;
        aSrcDst.w = alpha;
    }
};

template <RealVector T> struct NOP
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = aSrc1;
    }
    DEVICE_CODE void operator()(T & /*aSrcDst*/) const
    {
    }
};

template <RealVector T> struct ColorTwist3x3
{
    Matrix<remove_vector_t<T>> ColorMatrix;

    ColorTwist3x3(const Matrix<remove_vector_t<T>> &aColorMatrix) : ColorMatrix(aColorMatrix)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = ColorMatrix * aSrc1;
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = ColorMatrix * aSrcDst;
    }
};

template <RealVector T> struct ColorTwist3x4
{
    Matrix3x4<remove_vector_t<T>> ColorMatrix;

    ColorTwist3x4(const Matrix3x4<remove_vector_t<T>> &aColorMatrix) : ColorMatrix(aColorMatrix)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = ColorMatrix * aSrc1;
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = ColorMatrix * aSrcDst;
    }
};

template <RealVector T> struct ColorTwist4x4
{
    Matrix4x4<remove_vector_t<T>> ColorMatrix;

    ColorTwist4x4(const Matrix4x4<remove_vector_t<T>> &aColorMatrix) : ColorMatrix(aColorMatrix)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = ColorMatrix * aSrc1;
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = ColorMatrix * aSrcDst;
    }
};

template <RealVector T> struct ColorTwist4x4C
{
    Matrix4x4<remove_vector_t<T>> ColorMatrix;
    Vector4<remove_vector_t<T>> Constant;

    ColorTwist4x4C(const Matrix4x4<remove_vector_t<T>> &aColorMatrix, const Vector4<remove_vector_t<T>> &aConstant)
        : ColorMatrix(aColorMatrix), Constant(aConstant)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = Constant + ColorMatrix * aSrc1;
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = Constant + ColorMatrix * aSrcDst;
    }
};

// Linear to gamma corrected
template <RealVector T> struct GammaBT709
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    GammaBT709() : NormFactor(static_cast<TScalar>(1))
    {
    }
    GammaBT709(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        T res = aSrc / NormFactor;

        for (int i = 0; i < vector_active_size_v<T>; i++)
        {
            TScalar v = res[static_cast<Channel>(i)];

            if (v < 0.018f)
            {
                v *= 4.5f;
            }
            else
            {
                v = 1.099f * powf(v, 0.45f) - 0.099f;
            }

            res[static_cast<Channel>(i)] = v;
        }
        res.Min({1.0f});
        res.Max({0.0f});
        res *= NormFactor;

        return res;
    }
};

// Gamma corrected to linear
template <RealVector T> struct GammaInvBT709
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    GammaInvBT709() : NormFactor(static_cast<TScalar>(1))
    {
    }
    GammaInvBT709(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        T res = aSrc / NormFactor;

        for (int i = 0; i < vector_active_size_v<T>; i++)
        {
            TScalar v = res[static_cast<Channel>(i)];

            if (v < 0.0812f)
            {
                v /= 4.5f;
            }
            else
            {
                v = powf((v + 0.099f) / 1.099f, 2.22f);
            }

            res[static_cast<Channel>(i)] = v;
        }
        res.Min({1.0f});
        res.Max({0.0f});
        res *= NormFactor;

        return res;
    }
};

// Linear to gamma corrected
template <RealVector T> struct GammasRGB
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    GammasRGB() : NormFactor(static_cast<TScalar>(1))
    {
    }
    GammasRGB(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        T res = aSrc / NormFactor;

        for (int i = 0; i < vector_active_size_v<T>; i++)
        {
            TScalar v = res[static_cast<Channel>(i)];

            if (v < 0.0031308f)
            {
                v *= 12.92f;
            }
            else
            {
                v = 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
            }

            res[static_cast<Channel>(i)] = v;
        }
        res.Min({1.0f});
        res.Max({0.0f});
        res *= NormFactor;

        return res;
    }
};

// Gamma corrected to linear
template <RealVector T> struct GammaInvsRGB
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    GammaInvsRGB() : NormFactor(static_cast<TScalar>(1))
    {
    }
    GammaInvsRGB(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        T res = aSrc / NormFactor;

        for (int i = 0; i < vector_active_size_v<T>; i++)
        {
            TScalar v = res[static_cast<Channel>(i)];

            if (v < 0.04045f)
            {
                v /= 12.92f;
            }
            else
            {
                v = powf((v + 0.055f) / 1.055f, 2.4f);
            }

            res[static_cast<Channel>(i)] = v;
        }
        res.Min({1.0f});
        res.Max({0.0f});
        res *= NormFactor;

        return res;
    }
};

template <RealVector SrcT, Norm norm> struct ColorGradientToGray
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, Vector1<remove_vector_t<SrcT>> &aDst) const
        requires(norm == Norm::Inf)
    {
        if constexpr (RealSignedVector<SrcT>)
        {
            SrcT temp = SrcT::Abs(aSrc1);
            aDst      = temp.Max();
        }
        else
        {
            aDst = aSrc1.Max();
        }
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, Vector1<remove_vector_t<SrcT>> &aDst) const
        requires(norm == Norm::L1)
    {
        SrcT temp;
        if constexpr (RealSignedVector<SrcT>)
        {
            temp = SrcT::Abs(aSrc1);
        }
        else
        {
            temp = aSrc1;
        }
        aDst = temp.x;

        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            aDst += temp.y;
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            aDst += temp.z;
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            aDst += temp.w;
        }
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, Vector1<remove_vector_t<SrcT>> &aDst) const
        requires(norm == Norm::L2)
    {
        SrcT temp = SrcT::Sqr(aSrc1);
        aDst      = temp.x;

        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            aDst += temp.y;
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            aDst += temp.z;
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            aDst += temp.w;
        }
        aDst.Sqrt();
    }
};

template <RealVector SrcT> struct ColorToGray
{
    const SrcT weights;

    ColorToGray(SrcT aWeights) : weights(aWeights)
    {
    }

    DEVICE_CODE void operator()(const SrcT &aSrc1, Vector1<remove_vector_t<SrcT>> &aDst) const
    {
        SrcT temp = weights * aSrc1;
        aDst      = temp.x;

        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            aDst += temp.y;
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            aDst += temp.z;
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            aDst += temp.w;
        }
    }
};

template <RealVector T, bool doNormalize> struct RGBtoLUV
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    RGBtoLUV() : NormFactor(static_cast<TScalar>(1))
    {
    }
    RGBtoLUV(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        // as in IPP or NPP:
        T c = aSrc;
        if constexpr (doNormalize)
        {
            c /= NormFactor;
        }

        // convert RGB to XYZ:
        c = color::RGBtoXYZ * c;

        // use CIE D65 chromaticity coordinates (IPP directly provides un and vn)
        constexpr TScalar CIE_XYZ_D65_xn = 0.312713f;
        constexpr TScalar CIE_XYZ_D65_yn = 0.329016f;
        constexpr TScalar DIVISOR        = -2.0f * CIE_XYZ_D65_xn + 12.0f * CIE_XYZ_D65_yn + 3.0f;
        constexpr TScalar un             = 4.0f * CIE_XYZ_D65_xn / DIVISOR; // un = 0.197839 according to IPP
        constexpr TScalar vn             = 9.0f * CIE_XYZ_D65_yn / DIVISOR; // vn = 0.468342 according to IPP

        // Now calculate LUV from the XYZ value
        TScalar nTemp = c.x + 15.0f * c.y + 3.0f * c.z;
        TScalar nu    = 4.0f * c.x / nTemp;
        TScalar nv    = 9.0f * c.y / nTemp;
        TScalar nL    = 116.0f * cbrtf(c.y) - 16.0f;

#ifdef IS_HOST_COMPILER
        nL = std::max(0.0f, std::min(nL, 100.0f));
#else
        nL = max(0.0f, min(nL, 100.0f));
#endif

        nTemp      = 13.0f * nL;
        TScalar nU = nTemp * (nu - un);

#ifdef IS_HOST_COMPILER
        nU = std::max(-134.0f, std::min(nU, 220.0f));
#else
        nU = max(-134.0f, min(nU, 220.0f));
#endif

        TScalar nV = nTemp * (nv - vn);
#ifdef IS_HOST_COMPILER
        nV = std::max(-140.0f, std::min(nV, 122.0f));
#else
        nV = max(-140.0f, min(nV, 122.0f));
#endif
        const TScalar L = nL * NormFactor * 0.01f;                 // / 100.0f
        const TScalar U = (nU + 134.0f) * NormFactor * 0.0028249f; // / 354.0f
        const TScalar V = (nV + 140.0f) * NormFactor * 0.0038168f; // / 262.0f
        return {L, U, V};
    }
};

template <RealVector T, bool doNormalize> struct LUVtoRGB
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    LUVtoRGB() : NormFactor(static_cast<TScalar>(1))
    {
    }
    LUVtoRGB(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        // as in IPP or NPP:
        T c = aSrc;
        if constexpr (doNormalize)
        {
            c /= NormFactor;
        }

        // use CIE D65 chromaticity coordinates (IPP directly provides un and vn)
        constexpr TScalar CIE_XYZ_D65_xn = 0.312713f;
        constexpr TScalar CIE_XYZ_D65_yn = 0.329016f;
        constexpr TScalar DIVISOR        = -2.0f * CIE_XYZ_D65_xn + 12.0f * CIE_XYZ_D65_yn + 3.0f;
        constexpr TScalar un             = 4.0f * CIE_XYZ_D65_xn / DIVISOR; // un = 0.197839 according to IPP
        constexpr TScalar vn             = 9.0f * CIE_XYZ_D65_yn / DIVISOR; // vn = 0.468342 according to IPP

        // First convert normalized LUV back to original CIE LUV range
        const TScalar nL = c.x * 100.0f;
        const TScalar nU = (c.y * 354.0f) - 134.0f;
        const TScalar nV = (c.z * 262.0f) - 140.0f;

        // Now convert LUV to CIE XYZ
        TScalar nTemp = 13.0f * nL;
        TScalar nu    = nU / nTemp + un;
        TScalar nv    = nV / nTemp + vn;

        T xyz;
        if (nL > 7.9996248f)
        {
            xyz.y = (nL + 16.0f) * 0.008621f; // / 116.0f
            xyz.y = powf(xyz.y, 3.0f);
        }
        else
        {
            xyz.y = nL * 0.001107f; // / 903.3f
        }
        xyz.x = (-9.0f * xyz.y * nu) / ((nu - 4.0f) * nv - nu * nv);
        xyz.z = (9.0f * xyz.y - 15.0f * nv * xyz.y - nv * xyz.x) / (3.0f * nv);

        c = color::XYZtoRGB * xyz;

        c.Max({0.0f});
        c.Min({1.0f});

        return c;
    }
};

template <RealVector T, bool doNormalize> struct BGRtoLUV : public RGBtoLUV<T, doNormalize>
{
    BGRtoLUV()
    {
    }
    BGRtoLUV(RGBtoLUV<T, doNormalize>::TScalar aNormFactor) : RGBtoLUV<T, doNormalize>(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = RGBtoLUV<T, doNormalize>::compute(color::Swap1_3 * aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = RGBtoLUV<T, doNormalize>::compute(color::Swap1_3 * aSrcDst);
    }
};

template <RealVector T, bool doNormalize> struct LUVtoBGR : public LUVtoRGB<T, doNormalize>
{
    LUVtoBGR()
    {
    }
    LUVtoBGR(LUVtoRGB<T, doNormalize>::TScalar aNormFactor) : LUVtoRGB<T, doNormalize>(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = color::Swap1_3 * LUVtoRGB<T, doNormalize>::compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = color::Swap1_3 * LUVtoRGB<T, doNormalize>::compute(aSrcDst);
    }
};

template <RealVector T, bool doNormalize> struct RGBtoHLS
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    RGBtoHLS() : NormFactor(static_cast<TScalar>(1))
    {
    }
    RGBtoHLS(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        // as in IPP or NPP:
        T c = aSrc;
        if constexpr (doNormalize)
        {
            c /= NormFactor;
        }

        TScalar S = 0.0f;
        TScalar H = 0.0f;
        // Lightness
        TScalar max     = c.Max();
        TScalar min     = c.Min();
        TScalar L       = (max + min) * 0.5f;
        TScalar divisor = max - min;
        // Saturation
        if (divisor == 0.0f) // achromatic case
        {
            if constexpr (doNormalize)
            {
                L = L * NormFactor;
            }
            return {H, L, S};
        }

        // chromatic case
        if (L > 0.5f)
        {
            S = divisor / (1.0f - (max + min - 1.0f));
        }
        else
        {
            S = divisor / (max + min);
        }

        // Hue
        TScalar cr = (max - c.x) / divisor;
        TScalar cg = (max - c.y) / divisor;
        TScalar cb = (max - c.z) / divisor;
        if (c.x == max)
        {
            H = cb - cg;
        }
        else if (c.y == max)
        {
            H = 2.0f + cr - cb;
        }
        else if (c.z == max)
        {
            H = 4.0f + cg - cr;
        }
        H = H * 0.166667f; // / 6.0F
        if (H < 0.0f)
        {
            H = H + 1.0f;
        }

        if constexpr (doNormalize)
        {
            H = H * NormFactor;
            L = L * NormFactor;
            S = S * NormFactor;
        }

        return {H, L, S};
    }
};

template <RealVector T, bool doNormalize> struct HLStoRGB
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    HLStoRGB() : NormFactor(static_cast<TScalar>(1))
    {
    }
    HLStoRGB(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        // as in IPP or NPP:
        T c = aSrc;
        if constexpr (doNormalize)
        {
            c /= NormFactor;
        }

        TScalar m1;
        TScalar m2;
        TScalar r;
        TScalar g;
        TScalar b;
        TScalar h = 0.0f;
        if (c.y <= 0.5f)
        {
            m2 = c.y * (1.0f + c.z);
        }
        else
        {
            m2 = c.y + c.z - c.y * c.z;
        }
        m1 = 2.0f * c.y - m2;
        if (c.z == 0.0f)
        {
            r = g = b = c.y;
        }
        else
        {
            h = c.x + 0.3333f;
            if (h > 1.0f)
            {
                h -= 1.0f;
            }
        }
        TScalar mDiff = m2 - m1;
        if (0.6667f < h)
        {
            r = m1;
        }
        else
        {
            if (h < 0.1667f)
            {
                r = (m1 + mDiff * h * 6.0f); // / 0.1667f
            }
            else if (h < 0.5f)
            {
                r = m2;
            }
            else
            {
                r = m1 + mDiff * (0.6667f - h) * 6.0f; // / 0.1667f
            }
        }
#ifdef IS_HOST_COMPILER
        r = std::min(r, 1.0f);
#else
        r = min(r, 1.0f);
#endif

        h = c.x;
        if (0.6667f < h)
        {
            g = m1;
        }
        else
        {
            if (h < 0.1667f)
            {
                g = (m1 + mDiff * h * 6.0f); // / 0.1667f
            }
            else if (h < 0.5f)
            {
                g = m2;
            }
            else
            {
                g = m1 + mDiff * (0.6667f - h) * 6.0f; // / 0.1667f
            }
        }
#ifdef IS_HOST_COMPILER
        g = std::min(g, 1.0f);
#else
        g = min(g, 1.0f);
#endif
        h = c.x - 0.3333f;
        if (h < 0.0f)
        {
            h += 1.0f;
        }
        if (0.6667f < h)
        {
            b = m1;
        }
        else
        {
            if (h < 0.1667f)
            {
                b = (m1 + mDiff * h * 6.0f); // / 0.1667F
            }
            else if (h < 0.5f)
            {
                b = m2;
            }
            else
            {
                b = m1 + mDiff * (0.6667f - h) * 6.0f; // / 0.1667F
            }
        }
#ifdef IS_HOST_COMPILER
        b = std::min(b, 1.0f);
#else
        b = min(b, 1.0f);
#endif

        if constexpr (doNormalize)
        {
            r = r * NormFactor;
            g = g * NormFactor;
            b = b * NormFactor;
        }

        return {r, g, b};
    }
};

template <RealVector T, bool doNormalize> struct BGRtoHLS : public RGBtoHLS<T, doNormalize>
{
    BGRtoHLS()
    {
    }
    BGRtoHLS(RGBtoHLS<T, doNormalize>::TScalar aNormFactor) : RGBtoHLS<T, doNormalize>(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = RGBtoHLS<T, doNormalize>::compute(color::Swap1_3 * aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = RGBtoHLS<T, doNormalize>::compute(color::Swap1_3 * aSrcDst);
    }
};

template <RealVector T, bool doNormalize> struct HLStoBGR : public HLStoRGB<T, doNormalize>
{
    HLStoBGR()
    {
    }
    HLStoBGR(HLStoRGB<T, doNormalize>::TScalar &aNormFactor) : HLStoRGB<T, doNormalize>(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = color::Swap1_3 * HLStoRGB<T, doNormalize>::compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = color::Swap1_3 * HLStoRGB<T, doNormalize>::compute(aSrcDst);
    }
};

template <RealVector T, bool doNormalize> struct RGBtoLab
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    RGBtoLab() : NormFactor(static_cast<TScalar>(1))
    {
    }
    RGBtoLab(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        // as in IPP or NPP:
        T c = aSrc;
        if constexpr (doNormalize)
        {
            c /= NormFactor;
        }

        // use CIE Lab chromaticity coordinates
        constexpr TScalar CIE_LAB_D65_xn = 0.950455f;
        constexpr TScalar CIE_LAB_D65_zn = 1.088753f;

        T xyz = color::RGBtoXYZ * c;

        TScalar L = cbrtf(xyz.y);
        TScalar a;
        TScalar b;
        TScalar fX = xyz.x / CIE_LAB_D65_xn;
        TScalar fY;
        TScalar fZ = xyz.z / CIE_LAB_D65_zn;
        fY         = L - 16.0f;
        L          = 116.0f * L - 16.0f;
        a          = cbrtf(fX) - 16.0f;
        a          = 500.0f * (a - fY);
        b          = cbrtf(fZ) - 16.0f;
        b          = 200.0f * (fY - b);

        // Now scale Lab range
        if (NormFactor == 255.0f) // uchar data type
        {
            L = L * NormFactor * 0.01f; // / 100.0F
            a = a + 128.0f;
            b = b + 128.0f;
        }
        else if (NormFactor == 65535.0f) // ushort data type
        {
            L = L * NormFactor * 0.01f; // / 100.0F
            a = (a + 128.0f) * 255.0f;
            b = (b + 128.0f) * 255.0f;
        }

        return {L, a, b};
    }
};

template <RealVector T, bool doNormalize> struct LabtoRGB
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    LabtoRGB() : NormFactor(static_cast<TScalar>(1))
    {
    }
    LabtoRGB(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        // as in IPP or NPP:
        T c = aSrc;

        // First convert Lab back to original range then to CIE XYZ
        if constexpr (doNormalize)
        {
            if (NormFactor == 255.0f) // uchar data type
            {
                c.x = c.x * 100.0f / NormFactor;
                c.y = c.y - 128.0f;
                c.z = c.z - 128.0f;
            }
            else if (NormFactor == 65535.0f) // ushort data type
            {
                c.x = c.x * 100.0f / NormFactor;
                c.y = c.y / 255.0f - 128.0f;
                c.z = c.z / 255.0f - 128.0f;
            }
        }

        // use CIE Lab chromaticity coordinates
        constexpr TScalar CIE_LAB_D65_xn = 0.950455f;
        constexpr TScalar CIE_LAB_D65_zn = 1.088753f;

        // First convert Lab back to original range then to CIE XYZ

        TScalar nP = (c.x + 16.0f) * 0.008621f;                        // / 116.0f
        TScalar x  = nP * nP * nP;                                     // powf(nP, 3.0f);
        TScalar y  = CIE_LAB_D65_xn * powf((nP + c.y * 0.002f), 3.0f); // / 500.0f
        TScalar z  = CIE_LAB_D65_zn * powf((nP - c.z * 0.005f), 3.0f); // / 200.0f

        T rgb = color::XYZtoRGB * T(x, y, z);

        rgb.Min(T(1));

        if constexpr (doNormalize)
        {
            rgb *= NormFactor;
        }

        return rgb;
    }
};

template <RealVector T, bool doNormalize> struct BGRtoLab : public RGBtoLab<T, doNormalize>
{
    BGRtoLab()
    {
    }
    BGRtoLab(RGBtoLab<T, doNormalize>::TScalar aNormFactor) : RGBtoLab<T, doNormalize>(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = RGBtoLab<T, doNormalize>::compute(color::Swap1_3 * aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = RGBtoLab<T, doNormalize>::compute(color::Swap1_3 * aSrcDst);
    }
};

template <RealVector T, bool doNormalize> struct LabtoBGR : public LabtoRGB<T, doNormalize>
{
    LabtoBGR()
    {
    }
    LabtoBGR(LabtoRGB<T, doNormalize>::TScalar &aNormFactor) : LabtoRGB<T, doNormalize>(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = color::Swap1_3 * LabtoRGB<T, doNormalize>::compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = color::Swap1_3 * LabtoRGB<T, doNormalize>::compute(aSrcDst);
    }
};

template <RealVector T, bool doNormalize> struct RGBtoHSV
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    RGBtoHSV() : NormFactor(static_cast<TScalar>(1))
    {
    }
    RGBtoHSV(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        // as in IPP or NPP:
        T c = aSrc;
        if constexpr (doNormalize)
        {
            c /= NormFactor;
        }

        TScalar nS = 0.0f;
        TScalar nH = 0.0f;
        // Value
        TScalar nV = c.Max();
        // Saturation
        TScalar nTemp    = c.Min();
        TScalar nDivisor = nV - nTemp;

        if (nV == 0.0f) // achromatic case
        {
            return {0.0f, 0.0f, 0.0f};
        }

        if (nDivisor == 0.0f)
        {
            return {0.0f, 0.0f, nV * NormFactor};
        }

        // chromatic case
        nS = nDivisor / nV;

        // Hue:
        TScalar nCr = (nV - c.x) / nDivisor;
        TScalar nCg = (nV - c.y) / nDivisor;
        TScalar nCb = (nV - c.z) / nDivisor;
        if (c.x == nV)
        {
            nH = nCb - nCg;
        }
        else if (c.y == nV)
        {
            nH = 2.0f + nCr - nCb;
        }
        else if (c.z == nV)
        {
            nH = 4.0f + nCg - nCr;
        }
        nH = nH * 0.166667f; // / 6.0F
        if (nH < 0.0f)
        {
            nH = nH + 1.0f;
        }

        T ret(nH, nS, nV);
        if constexpr (doNormalize)
        {
            ret *= NormFactor;
        }

        return ret;
    }
};

template <RealVector T, bool doNormalize> struct HSVtoRGB
{
    using TScalar = remove_vector_t<T>;
    TScalar NormFactor;

    HSVtoRGB() : NormFactor(static_cast<TScalar>(1))
    {
    }
    HSVtoRGB(TScalar aNormFactor) : NormFactor(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = compute(aSrcDst);
    }

  protected:
    DEVICE_CODE T compute(const T &aSrc) const
    {
        // as in IPP or NPP:
        T c = aSrc;

        // First convert Lab back to original range then to CIE XYZ
        if constexpr (doNormalize)
        {
            c /= NormFactor;
        }

        TScalar nR = 0.0f;
        TScalar nG = 0.0f;
        TScalar nB = 0.0f;
        if (c.y == 0.0f)
        {
            c.x = c.y = c.z;
            if constexpr (doNormalize)
            {
                c *= NormFactor;
            }
            return c;
        }

        if (c.x == 1.0f)
        {
            c.x = 0.0f;
        }
        else
        {
            c.x = c.x * 6.0f; // / 0.1667f
        }

        TScalar nI = floorf(c.x);
        TScalar nF = c.x - nI;
        TScalar nM = c.z * (1.0f - c.y);
        TScalar nN = c.z * (1.0f - c.y * nF);
        TScalar nK = c.z * (1.0f - c.y * (1.0f - nF));
        if (nI == 0.0f)
        {
            nR = c.z;
            nG = nK;
            nB = nM;
        }
        else if (nI == 1.0f)
        {
            nR = nN;
            nG = c.z;
            nB = nM;
        }
        else if (nI == 2.0f)
        {
            nR = nM;
            nG = c.z;
            nB = nK;
        }
        else if (nI == 3.0F)
        {
            nR = nM;
            nG = nN;
            nB = c.z;
        }
        else if (nI == 4.0F)
        {
            nR = nK;
            nG = nM;
            nB = c.z;
        }
        else if (nI == 5.0F)
        {
            nR = c.z;
            nG = nM;
            nB = nN;
        }

        c.x = nR;
        c.y = nG;
        c.z = nB;
        if constexpr (doNormalize)
        {
            c *= NormFactor;
        }

        return c;
    }
};

template <RealVector T, bool doNormalize> struct BGRtoHSV : public RGBtoHSV<T, doNormalize>
{
    BGRtoHSV()
    {
    }
    BGRtoHSV(RGBtoHSV<T, doNormalize>::TScalar aNormFactor) : RGBtoHSV<T, doNormalize>(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = RGBtoHSV<T, doNormalize>::compute(color::Swap1_3 * aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = RGBtoHSV<T, doNormalize>::compute(color::Swap1_3 * aSrcDst);
    }
};

template <RealVector T, bool doNormalize> struct HSVtoBGR : public HSVtoRGB<T, doNormalize>
{
    HSVtoBGR()
    {
    }
    HSVtoBGR(HSVtoRGB<T, doNormalize>::TScalar &aNormFactor) : HSVtoRGB<T, doNormalize>(aNormFactor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = color::Swap1_3 * HSVtoRGB<T, doNormalize>::compute(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = color::Swap1_3 * HSVtoRGB<T, doNormalize>::compute(aSrcDst);
    }
};

template <AnyVector SrcT, AnyVector LutT, AnyVector DstT> struct LUTPaletteWithBounds
{
    const LutT *RESTRICT SrcPointer0;
    const int IndexBound;
    static constexpr int Offset = static_cast<int>(mpp::numeric_limits<remove_vector_t<SrcT>>::lowest());

    LUTPaletteWithBounds(const LutT *RESTRICT aSrcPointer0, int aIndexBound)
        : SrcPointer0(aSrcPointer0), IndexBound(aIndexBound)
    {
    }

    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst) const
    {
        const int idx = static_cast<int>(aSrc1.x) + Offset;
        if (idx < IndexBound)
        {
            // This is for 4 element aligned 3 element LUTs:
            if constexpr (vector_size_v<LutT> == 4 && vector_size_v<DstT> == 3)
            {
                aDst = SrcPointer0[idx].XYZ();
            }
            else
            {
                aDst = SrcPointer0[idx];
            }
        }
    }
    DEVICE_CODE void operator()(DstT &aSrcDst) const
        requires(std::same_as<DstT, LutT> && std::same_as<DstT, SrcT>)
    {
        const int idx = static_cast<int>(aSrcDst.x) + Offset;
        if (idx < IndexBound)
        {
            aSrcDst = SrcPointer0[idx];
        }
    }
};

template <AnyVector T> struct LUTPalettePlanar2WithBounds
{
    const remove_vector_t<T> *RESTRICT SrcPointer0;
    const remove_vector_t<T> *RESTRICT SrcPointer1;
    const int IndexBound;
    static constexpr int Offset = static_cast<int>(numeric_limits<remove_vector_t<T>>::lowest());

    LUTPalettePlanar2WithBounds(const remove_vector_t<T> *RESTRICT aSrcPointer0,
                                const remove_vector_t<T> *RESTRICT aSrcPointer1, int aIndexBound)
        : SrcPointer0(aSrcPointer0), SrcPointer1(aSrcPointer1), IndexBound(aIndexBound)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        const int idx0 = static_cast<int>(aSrc1.x) + Offset;
        if (idx0 < IndexBound)
        {
            aDst.x = SrcPointer0[idx0];
        }
        const int idx1 = static_cast<int>(aSrc1.y) + Offset;
        if (idx1 < IndexBound)
        {
            aDst.y = SrcPointer1[idx1];
        }
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        const int idx0 = static_cast<int>(aSrcDst.x) + Offset;
        if (idx0 < IndexBound)
        {
            aSrcDst.x = SrcPointer0[idx0];
        }
        const int idx1 = static_cast<int>(aSrcDst.y) + Offset;
        if (idx1 < IndexBound)
        {
            aSrcDst.y = SrcPointer1[idx1];
        }
    }
};

template <AnyVector T> struct LUTPalettePlanar3WithBounds
{
    const remove_vector_t<T> *RESTRICT SrcPointer0;
    const remove_vector_t<T> *RESTRICT SrcPointer1;
    const remove_vector_t<T> *RESTRICT SrcPointer2;
    const int IndexBound;
    static constexpr int Offset = static_cast<int>(numeric_limits<remove_vector_t<T>>::lowest());

    LUTPalettePlanar3WithBounds(const remove_vector_t<T> *RESTRICT aSrcPointer0,
                                const remove_vector_t<T> *RESTRICT aSrcPointer1,
                                const remove_vector_t<T> *RESTRICT aSrcPointer2, int aIndexBound)
        : SrcPointer0(aSrcPointer0), SrcPointer1(aSrcPointer1), SrcPointer2(aSrcPointer2), IndexBound(aIndexBound)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        const int idx0 = static_cast<int>(aSrc1.x) + Offset;
        if (idx0 < IndexBound)
        {
            aDst.x = SrcPointer0[idx0];
        }
        const int idx1 = static_cast<int>(aSrc1.y) + Offset;
        if (idx1 < IndexBound)
        {
            aDst.y = SrcPointer1[idx1];
        }
        const int idx2 = static_cast<int>(aSrc1.z) + Offset;
        if (idx2 < IndexBound)
        {
            aDst.z = SrcPointer2[idx2];
        }
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        const int idx0 = static_cast<int>(aSrcDst.x) + Offset;
        if (idx0 < IndexBound)
        {
            aSrcDst.x = SrcPointer0[idx0];
        }
        const int idx1 = static_cast<int>(aSrcDst.y) + Offset;
        if (idx1 < IndexBound)
        {
            aSrcDst.y = SrcPointer1[idx1];
        }
        const int idx2 = static_cast<int>(aSrcDst.z) + Offset;
        if (idx2 < IndexBound)
        {
            aSrcDst.z = SrcPointer2[idx2];
        }
    }
};

template <AnyVector T> struct LUTPalettePlanar4WithBounds
{
    const remove_vector_t<T> *RESTRICT SrcPointer0;
    const remove_vector_t<T> *RESTRICT SrcPointer1;
    const remove_vector_t<T> *RESTRICT SrcPointer2;
    const remove_vector_t<T> *RESTRICT SrcPointer3;
    const int IndexBound;
    static constexpr int Offset = static_cast<int>(numeric_limits<remove_vector_t<T>>::lowest());

    LUTPalettePlanar4WithBounds(const remove_vector_t<T> *RESTRICT aSrcPointer0,
                                const remove_vector_t<T> *RESTRICT aSrcPointer1,
                                const remove_vector_t<T> *RESTRICT aSrcPointer2,
                                const remove_vector_t<T> *RESTRICT aSrcPointer3, int aIndexBound)
        : SrcPointer0(aSrcPointer0), SrcPointer1(aSrcPointer1), SrcPointer2(aSrcPointer2), SrcPointer3(aSrcPointer3),
          IndexBound(aIndexBound)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        const int idx0 = static_cast<int>(aSrc1.x) + Offset;
        if (idx0 < IndexBound)
        {
            aDst.x = SrcPointer0[idx0];
        }
        const int idx1 = static_cast<int>(aSrc1.y) + Offset;
        if (idx1 < IndexBound)
        {
            aDst.y = SrcPointer1[idx1];
        }
        const int idx2 = static_cast<int>(aSrc1.z) + Offset;
        if (idx2 < IndexBound)
        {
            aDst.z = SrcPointer2[idx2];
        }
        const int idx3 = static_cast<int>(aSrc1.w) + Offset;
        if (idx3 < IndexBound)
        {
            aDst.w = SrcPointer3[idx3];
        }
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        const int idx0 = static_cast<int>(aSrcDst.x) + Offset;
        if (idx0 < IndexBound)
        {
            aSrcDst.x = SrcPointer0[idx0];
        }
        const int idx1 = static_cast<int>(aSrcDst.y) + Offset;
        if (idx1 < IndexBound)
        {
            aSrcDst.y = SrcPointer1[idx1];
        }
        const int idx2 = static_cast<int>(aSrcDst.z) + Offset;
        if (idx2 < IndexBound)
        {
            aSrcDst.z = SrcPointer2[idx2];
        }
        const int idx3 = static_cast<int>(aSrcDst.w) + Offset;
        if (idx3 < IndexBound)
        {
            aSrcDst.w = SrcPointer3[idx3];
        }
    }
};

template <AnyVector SrcT, AnyVector LutT, AnyVector DstT> struct LUTPalette
{
    const LutT *RESTRICT SrcPointer0;
    static constexpr int Offset = static_cast<int>(numeric_limits<remove_vector_t<SrcT>>::lowest());

    LUTPalette(const LutT *RESTRICT aSrcPointer0) : SrcPointer0(aSrcPointer0)
    {
    }

    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst) const
    {
        // This is for 4 element aligned 3 element LUTs:
        if constexpr (vector_size_v<LutT> == 4 && vector_size_v<DstT> == 3)
        {
            aDst = SrcPointer0[static_cast<int>(aSrc1.x) + Offset].XYZ();
        }
        else
        {
            aDst = SrcPointer0[static_cast<int>(aSrc1.x) + Offset];
        }
    }
    DEVICE_CODE void operator()(DstT &aSrcDst) const
        requires(std::same_as<DstT, LutT> && std::same_as<DstT, SrcT>)
    {
        aSrcDst = SrcPointer0[static_cast<int>(aSrcDst.x) + Offset];
    }
};

template <AnyVector T> struct LUTPalettePlanar2
{
    const remove_vector_t<T> *RESTRICT SrcPointer0;
    const remove_vector_t<T> *RESTRICT SrcPointer1;
    static constexpr int Offset = static_cast<int>(numeric_limits<remove_vector_t<T>>::lowest());

    LUTPalettePlanar2(const remove_vector_t<T> *RESTRICT aSrcPointer0, const remove_vector_t<T> *RESTRICT aSrcPointer1)
        : SrcPointer0(aSrcPointer0), SrcPointer1(aSrcPointer1)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = SrcPointer0[static_cast<int>(aSrc1.x) + Offset];
        aDst.y = SrcPointer1[static_cast<int>(aSrc1.y) + Offset];
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = SrcPointer0[static_cast<int>(aSrcDst.x) + Offset];
        aSrcDst.y = SrcPointer1[static_cast<int>(aSrcDst.y) + Offset];
    }
};

template <AnyVector T> struct LUTPalettePlanar3
{
    const remove_vector_t<T> *RESTRICT SrcPointer0;
    const remove_vector_t<T> *RESTRICT SrcPointer1;
    const remove_vector_t<T> *RESTRICT SrcPointer2;
    static constexpr int Offset = static_cast<int>(numeric_limits<remove_vector_t<T>>::lowest());

    LUTPalettePlanar3(const remove_vector_t<T> *RESTRICT aSrcPointer0, const remove_vector_t<T> *RESTRICT aSrcPointer1,
                      const remove_vector_t<T> *RESTRICT aSrcPointer2)
        : SrcPointer0(aSrcPointer0), SrcPointer1(aSrcPointer1), SrcPointer2(aSrcPointer2)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = SrcPointer0[static_cast<int>(aSrc1.x) + Offset];
        aDst.y = SrcPointer1[static_cast<int>(aSrc1.y) + Offset];
        aDst.z = SrcPointer2[static_cast<int>(aSrc1.z) + Offset];
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = SrcPointer0[static_cast<int>(aSrcDst.x) + Offset];
        aSrcDst.y = SrcPointer1[static_cast<int>(aSrcDst.y) + Offset];
        aSrcDst.z = SrcPointer2[static_cast<int>(aSrcDst.z) + Offset];
    }
};

template <AnyVector T> struct LUTPalettePlanar4
{
    const remove_vector_t<T> *RESTRICT SrcPointer0;
    const remove_vector_t<T> *RESTRICT SrcPointer1;
    const remove_vector_t<T> *RESTRICT SrcPointer2;
    const remove_vector_t<T> *RESTRICT SrcPointer3;
    static constexpr int Offset = static_cast<int>(numeric_limits<remove_vector_t<T>>::lowest());

    LUTPalettePlanar4(const remove_vector_t<T> *RESTRICT aSrcPointer0, const remove_vector_t<T> *RESTRICT aSrcPointer1,
                      const remove_vector_t<T> *RESTRICT aSrcPointer2, const remove_vector_t<T> *RESTRICT aSrcPointer3)
        : SrcPointer0(aSrcPointer0), SrcPointer1(aSrcPointer1), SrcPointer2(aSrcPointer2), SrcPointer3(aSrcPointer3)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = SrcPointer0[static_cast<int>(aSrc1.x) + Offset];
        aDst.y = SrcPointer1[static_cast<int>(aSrc1.y) + Offset];
        aDst.z = SrcPointer2[static_cast<int>(aSrc1.z) + Offset];
        aDst.w = SrcPointer3[static_cast<int>(aSrc1.w) + Offset];
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = SrcPointer0[static_cast<int>(aSrcDst.x) + Offset];
        aSrcDst.y = SrcPointer1[static_cast<int>(aSrcDst.y) + Offset];
        aSrcDst.z = SrcPointer2[static_cast<int>(aSrcDst.z) + Offset];
        aSrcDst.w = SrcPointer3[static_cast<int>(aSrcDst.w) + Offset];
    }
};

template <typename T, typename LUTT, typename ComputeT, InterpolationMode interpolationMode> struct LUTChannelSrc
{
    const LUTT *RESTRICT LutX;
    const LUTT *RESTRICT LutY;
    const int *RESTRICT Accelerator;
    const int LUTSize;
    const int AcceleratorSize;

    LUTChannelSrc(const LUTT *RESTRICT aLutX, const LUTT *RESTRICT aLutY, const int *RESTRICT aAccelerator,
                  int aLUTSize, int aAcceleratorSize)
        : LutX(aLutX), LutY(aLutY), Accelerator(aAccelerator), LUTSize(aLUTSize), AcceleratorSize(aAcceleratorSize)
    {
    }

    DEVICE_CODE T GetValue(T aX) const
    {
        if (aX < LutX[0] || aX >= LutX[LUTSize - 1])
        {
            return aX;
        }

        const float LutXRange = static_cast<float>(LutX[LUTSize - 1]) - static_cast<float>(LutX[0]);
        const float iAcc      = (static_cast<float>(aX) - static_cast<float>(LutX[0])) / LutXRange *
                           static_cast<float>(AcceleratorSize - 1);

        const int iStart = Accelerator[static_cast<int>(iAcc)];

        int idx = LastIndexSmallerOrEqual(LutX, LUTSize, iStart, static_cast<LUTT>(aX));

        if (idx >= 0)
        {
            if constexpr (interpolationMode == InterpolationMode::NearestNeighbor)
            {
                return static_cast<T>(LutY[idx]);
            }
            else if constexpr (interpolationMode == InterpolationMode::Linear)
            {
                using namespace std;
                idx = min(max(0, idx), LUTSize - 2);

                ComputeT x[2];
                ComputeT y[2];
                x[0] = static_cast<ComputeT>(*(LutX + idx + 0));
                x[1] = static_cast<ComputeT>(*(LutX + idx + 1));
                y[0] = static_cast<ComputeT>(*(LutY + idx + 0));
                y[1] = static_cast<ComputeT>(*(LutY + idx + 1));

                return static_cast<T>(LinearInterpolate(x, y, static_cast<ComputeT>(aX)));
            }
            else if constexpr (interpolationMode == InterpolationMode::CubicLagrange)
            {
                using namespace std;
                idx--;
                idx = min(max(0, idx), LUTSize - 4);

                ComputeT x[4];
                ComputeT y[4];
                x[0] = static_cast<ComputeT>(*(LutX + idx + 0));
                x[1] = static_cast<ComputeT>(*(LutX + idx + 1));
                x[2] = static_cast<ComputeT>(*(LutX + idx + 2));
                x[3] = static_cast<ComputeT>(*(LutX + idx + 3));
                y[0] = static_cast<ComputeT>(*(LutY + idx + 0));
                y[1] = static_cast<ComputeT>(*(LutY + idx + 1));
                y[2] = static_cast<ComputeT>(*(LutY + idx + 2));
                y[3] = static_cast<ComputeT>(*(LutY + idx + 3));

                return static_cast<T>(CubicInterpolate(x, y, static_cast<ComputeT>(aX)));
            }
        }
        // we should never end up here due to the initial bound checks...
        return aX;
    }
};

template <AnyVector T, typename LUTT, typename ComputeT, InterpolationMode interpolationMode> struct LUT1Channel
{
    using LUTChannelSrc_type = LUTChannelSrc<remove_vector_t<T>, LUTT, ComputeT, interpolationMode>;
    const LUTChannelSrc_type Channel0;

    LUT1Channel(const LUTChannelSrc_type &aChannel0) : Channel0(aChannel0)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = Channel0.GetValue(aSrc1.x);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = Channel0.GetValue(aSrcDst.x);
    }
};

template <AnyVector T, typename LUTT, typename ComputeT, InterpolationMode interpolationMode> struct LUT2Channel
{
    using LUTChannelSrc_type = LUTChannelSrc<remove_vector_t<T>, LUTT, ComputeT, interpolationMode>;
    const LUTChannelSrc_type Channel0;
    const LUTChannelSrc_type Channel1;

    LUT2Channel(const LUTChannelSrc_type &aChannel0, const LUTChannelSrc_type &aChannel1)
        : Channel0(aChannel0), Channel1(aChannel1)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = Channel0.GetValue(aSrc1.x);
        aDst.y = Channel1.GetValue(aSrc1.y);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = Channel0.GetValue(aSrcDst.x);
        aSrcDst.y = Channel1.GetValue(aSrcDst.y);
    }
};

template <AnyVector T, typename LUTT, typename ComputeT, InterpolationMode interpolationMode> struct LUT3Channel
{
    using LUTChannelSrc_type = LUTChannelSrc<remove_vector_t<T>, LUTT, ComputeT, interpolationMode>;
    const LUTChannelSrc_type Channel0;
    const LUTChannelSrc_type Channel1;
    const LUTChannelSrc_type Channel2;

    LUT3Channel(const LUTChannelSrc_type &aChannel0, const LUTChannelSrc_type &aChannel1,
                const LUTChannelSrc_type &aChannel2)
        : Channel0(aChannel0), Channel1(aChannel1), Channel2(aChannel2)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = Channel0.GetValue(aSrc1.x);
        aDst.y = Channel1.GetValue(aSrc1.y);
        aDst.z = Channel2.GetValue(aSrc1.z);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = Channel0.GetValue(aSrcDst.x);
        aSrcDst.y = Channel1.GetValue(aSrcDst.y);
        aSrcDst.z = Channel2.GetValue(aSrcDst.z);
    }
};

template <AnyVector T, typename LUTT, typename ComputeT, InterpolationMode interpolationMode> struct LUT4Channel
{
    using LUTChannelSrc_type = LUTChannelSrc<remove_vector_t<T>, LUTT, ComputeT, interpolationMode>;
    const LUTChannelSrc_type Channel0;
    const LUTChannelSrc_type Channel1;
    const LUTChannelSrc_type Channel2;
    const LUTChannelSrc_type Channel3;

    LUT4Channel(const LUTChannelSrc_type &aChannel0, const LUTChannelSrc_type &aChannel1,
                const LUTChannelSrc_type &aChannel2, const LUTChannelSrc_type &aChannel3)
        : Channel0(aChannel0), Channel1(aChannel1), Channel2(aChannel2), Channel3(aChannel3)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = Channel0.GetValue(aSrc1.x);
        aDst.y = Channel1.GetValue(aSrc1.y);
        aDst.z = Channel2.GetValue(aSrc1.z);
        aDst.w = Channel3.GetValue(aSrc1.w);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = Channel0.GetValue(aSrcDst.x);
        aSrcDst.y = Channel1.GetValue(aSrcDst.y);
        aSrcDst.z = Channel2.GetValue(aSrcDst.z);
        aSrcDst.w = Channel3.GetValue(aSrcDst.w);
    }
};

// Note: as linear interpolation with textures on GPU has only 8-bit resolution, this would only give good results for
// Pixel8uC4. To keep things simple, we will always use tri-linear interpolation calculated manually and not in the
// texture fetch units for all types for all devices.
template <AnyVector T, AnyVector LutType> struct LUTTrilinear
{
    const LutType *RESTRICT Lut3D;
    Vector3<float> MinRange;
    Vector3<float> TotalRange;
    Vector3<float> LutSizeMinus1;
    Vector3<int> LutSize;

    LUTTrilinear(const LutType *aLut3D, const Vector3<remove_vector_t<T>> &aMinRange,
                 const Vector3<remove_vector_t<T>> &aTotalRange, const Vector3<int> &aLutSize)
        : Lut3D(aLut3D), MinRange(aMinRange), TotalRange(aTotalRange), LutSizeMinus1(aLutSize - 1), LutSize(aLutSize)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        Vector3<float> coord;
        if constexpr (vector_size_v<T> == 4)
        {
            coord = Vector3<float>(aSrc1.XYZ());
        }
        else
        {
            coord = Vector3<float>(aSrc1);
        }
        coord = (coord - MinRange) / TotalRange * LutSizeMinus1;
        coord.Max({0.0f});
        coord.Min(LutSizeMinus1);
        Vector3<float> coordLow  = Vector3<float>::Floor(coord);
        Vector3<float> coordHigh = coordLow + 1;
        coordHigh.Min(LutSizeMinus1);

        Vec3i cMin = coordLow;
        Vec3i cMax = coordHigh;
        cMin.y *= LutSize.x;
        cMax.y *= LutSize.x;
        cMin.z *= LutSize.x * LutSize.y;
        cMax.z *= LutSize.x * LutSize.y;

        Pixel32fC3 p000 = Pixel32fC4A(*(Lut3D + cMin.z + cMin.y + cMin.x)).XYZ();
        Pixel32fC3 p001 = Pixel32fC4A(*(Lut3D + cMin.z + cMin.y + cMax.x)).XYZ();

        Pixel32fC3 p010 = Pixel32fC4A(*(Lut3D + cMin.z + cMax.y + cMin.x)).XYZ();
        Pixel32fC3 p011 = Pixel32fC4A(*(Lut3D + cMin.z + cMax.y + cMax.x)).XYZ();

        Pixel32fC3 p100 = Pixel32fC4A(*(Lut3D + cMax.z + cMin.y + cMin.x)).XYZ();
        Pixel32fC3 p101 = Pixel32fC4A(*(Lut3D + cMax.z + cMin.y + cMax.x)).XYZ();

        Pixel32fC3 p110 = Pixel32fC4A(*(Lut3D + cMax.z + cMax.y + cMin.x)).XYZ();
        Pixel32fC3 p111 = Pixel32fC4A(*(Lut3D + cMax.z + cMax.y + cMax.x)).XYZ();

        coord -= coordLow;

        p000 *= 1.0f - coord.x;
        p010 *= 1.0f - coord.x;
        p100 *= 1.0f - coord.x;
        p110 *= 1.0f - coord.x;

        p001 *= coord.x;
        p011 *= coord.x;
        p101 *= coord.x;
        p111 *= coord.x;

        p000 += p001;
        p010 += p011;
        p100 += p101;
        p110 += p111;

        p000 *= 1.0f - coord.y;
        p100 *= 1.0f - coord.y;
        p010 *= coord.y;
        p110 *= coord.y;

        p000 += p010;
        p100 += p110;

        p000 = p000 * (1.0f - coord.z) + p100 * coord.z;

        if constexpr (RealIntVector<T>)
        {
            p000.Round();
        }

        aDst.x = static_cast<remove_vector_t<T>>(p000.x);
        aDst.y = static_cast<remove_vector_t<T>>(p000.y);
        aDst.z = static_cast<remove_vector_t<T>>(p000.z);

        if constexpr (vector_size_v<T> == 4)
        {
            // copy alpha channel (for C4A types it will be overwritten in the kernel)
            aDst.w = aSrc1.w;
        }
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        (*this)(aSrcDst, aSrcDst);
    }
};
} // namespace mpp::image
