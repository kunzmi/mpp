#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/mpp_defs.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace mpp
{

template <RealVector T> struct MinRed
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.Min(aSrc1);
    }
    DEVICE_CODE void operator()(const remove_vector_t<T> &aSrc1, remove_vector_t<T> &aSrcDst) const
        requires DeviceCode<T> && (!Is16BitFloat<remove_vector_t<T>>)
    {
        aSrcDst = min(aSrcDst, aSrc1);
    }
    DEVICE_CODE void operator()(const remove_vector_t<T> &aSrc1, remove_vector_t<T> &aSrcDst) const
        requires DeviceCode<T> && Is16BitFloat<remove_vector_t<T>>
    {
        aSrcDst.Min(aSrc1);
    }
    void operator()(const remove_vector_t<T> &aSrc1, remove_vector_t<T> &aSrcDst) const
        requires HostCode<T>
    {
        aSrcDst = std::min(aSrcDst, aSrc1);
    }
};

template <RealVector T> struct MaxRed
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst) const
    {
        aSrcDst.Max(aSrc1);
    }
    DEVICE_CODE void operator()(const remove_vector_t<T> &aSrc1, remove_vector_t<T> &aSrcDst) const
        requires DeviceCode<T>
    {
        aSrcDst = max(aSrcDst, aSrc1);
    }
    DEVICE_CODE void operator()(const remove_vector_t<T> &aSrc1, remove_vector_t<T> &aSrcDst) const
        requires DeviceCode<T> && Is16BitFloat<remove_vector_t<T>>
    {
        aSrcDst.Max(aSrc1);
    }
    void operator()(const remove_vector_t<T> &aSrc1, remove_vector_t<T> &aSrcDst) const
        requires HostCode<T>
    {
        aSrcDst = std::max(aSrcDst, aSrc1);
    }
};

template <RealVector SrcT> struct MinIdx
{
    DEVICE_CODE void operator()(const SrcT &aSrc, int aX, SrcT &aDst,
                                same_vector_size_different_type_t<SrcT, int> &aResX) const
    {
        if (aSrc.x < aDst.x)
        {
            aDst.x  = aSrc.x;
            aResX.x = aX;
        }
        else if (aSrc.x == aDst.x && aX < aResX.x)
        {
            aResX.x = aX;
        }
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            if (aSrc.y < aDst.y)
            {
                aDst.y  = aSrc.y;
                aResX.y = aX;
            }
            else if (aSrc.y == aDst.y && aX < aResX.y)
            {
                aResX.y = aX;
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            if (aSrc.z < aDst.z)
            {
                aDst.z  = aSrc.z;
                aResX.z = aX;
            }
            else if (aSrc.z == aDst.z && aX < aResX.z)
            {
                aResX.z = aX;
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            if (aSrc.w < aDst.w)
            {
                aDst.w  = aSrc.w;
                aResX.w = aX;
            }
            else if (aSrc.w == aDst.w && aX < aResX.w)
            {
                aResX.w = aX;
            }
        }
    }
    DEVICE_CODE void operator()(const remove_vector_t<SrcT> &aSrc, int aX, remove_vector_t<SrcT> &aSrcDst,
                                int &aResX) const
    {
        if (aSrc < aSrcDst)
        {
            aSrcDst = aSrc;
            aResX   = aX;
        }
        else if (aSrc == aSrcDst && aX < aResX)
        {
            aResX = aX;
        }
    }
};

template <RealVector SrcT> struct MaxIdx
{
    DEVICE_CODE void operator()(const SrcT &aSrc, int aX, SrcT &aDst,
                                same_vector_size_different_type_t<SrcT, int> &aResX) const
    {
        if (aSrc.x > aDst.x)
        {
            aDst.x  = aSrc.x;
            aResX.x = aX;
        }
        else if (aSrc.x == aDst.x && aX < aResX.x)
        {
            aResX.x = aX;
        }
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            if (aSrc.y > aDst.y)
            {
                aDst.y  = aSrc.y;
                aResX.y = aX;
            }
            else if (aSrc.y == aDst.y && aX < aResX.y)
            {
                aResX.y = aX;
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            if (aSrc.z > aDst.z)
            {
                aDst.z  = aSrc.z;
                aResX.z = aX;
            }
            else if (aSrc.z == aDst.z && aX < aResX.z)
            {
                aResX.z = aX;
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            if (aSrc.w > aDst.w)
            {
                aDst.w  = aSrc.w;
                aResX.w = aX;
            }
            else if (aSrc.w == aDst.w && aX < aResX.w)
            {
                aResX.w = aX;
            }
        }
    }
    DEVICE_CODE void operator()(const remove_vector_t<SrcT> &aSrc, int aX, remove_vector_t<SrcT> &aSrcDst,
                                int &aResX) const
    {
        if (aSrc > aSrcDst)
        {
            aSrcDst = aSrc;
            aResX   = aX;
        }
        else if (aSrc == aSrcDst && aX < aResX)
        {
            aResX = aX;
        }
    }
};

template <RealVector SrcT> struct CountInRange
{
    SrcT lowerLimit;
    SrcT upperLimit;

    CountInRange(const SrcT &aLowerLimit, const SrcT &aUpperLimit) : lowerLimit(aLowerLimit), upperLimit(aUpperLimit)
    {
    }

    DEVICE_CODE void operator()(const SrcT &aSrc, same_vector_size_different_type_t<SrcT, ulong64> &aDst) const
    {
        if (aSrc.x >= lowerLimit.x && aSrc.x <= upperLimit.x)
        {
            aDst.x++;
        }
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            if (aSrc.y >= lowerLimit.y && aSrc.y <= upperLimit.y)
            {
                aDst.y++;
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            if (aSrc.z >= lowerLimit.z && aSrc.z <= upperLimit.z)
            {
                aDst.z++;
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            if (aSrc.w >= lowerLimit.w && aSrc.w <= upperLimit.w)
            {
                aDst.w++;
            }
        }
    }
};

template <AnyVector SrcT, AnyVector DstT> struct Sum
{
    DEVICE_CODE void operator()(const SrcT &aSrc, DstT &aDst) const
    {
        aDst += DstT(aSrc);
    }
    DEVICE_CODE void operator()(const SrcT &aSrc, DstT &aDst) const
        requires std::same_as<SrcT, DstT>
    {
        aDst += aSrc;
    }
    DEVICE_CODE void operator()(const complex_basetype_t<remove_vector_t<SrcT>> &aSrc,
                                complex_basetype_t<remove_vector_t<DstT>> &aDst) const
    {
        aDst += complex_basetype_t<remove_vector_t<DstT>>(aSrc);
    }
};

template <AnyVector SrcT, AnyVector DstT, int Select> struct Sum1Or2
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        if constexpr (Select == 1)
        {
            aDst += DstT(aSrc1);
        }
        else
        {
            aDst += DstT(aSrc2);
        }
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires std::same_as<SrcT, DstT>
    {
        if constexpr (Select == 1)
        {
            aDst += aSrc1;
        }
        else
        {
            aDst += aSrc2;
        }
    }
};

template <AnyVector SrcT, AnyVector DstT> struct SumSqr
{
    DEVICE_CODE void operator()(const SrcT &aSrc, DstT &aDst) const
    {
        DstT temp(aSrc);
        temp.Sqr();
        aDst += temp;
    }
    // we use SumSqr to compute the standard deviation for complex numbers as for real numbers. For complex numbers we
    // need real and imag component squared seperately and not a complex operation:
    DEVICE_CODE void operator()(const SrcT &aSrc, DstT &aDst) const
        requires ComplexVector<SrcT> && ComplexVector<DstT>
    {
        DstT temp(aSrc);

        temp.x.real = temp.x.real * temp.x.real;
        temp.x.imag = temp.x.imag * temp.x.imag;

        if constexpr (vector_active_size_v<DstT> > 1)
        {
            temp.y.real = temp.y.real * temp.y.real;
            temp.y.imag = temp.y.imag * temp.y.imag;
        }
        if constexpr (vector_active_size_v<DstT> > 2)
        {
            temp.z.real = temp.z.real * temp.z.real;
            temp.z.imag = temp.z.imag * temp.z.imag;
        }
        if constexpr (vector_active_size_v<DstT> > 3)
        {
            temp.w.real = temp.w.real * temp.w.real;
            temp.w.imag = temp.w.imag * temp.w.imag;
        }

        aDst += temp;
    }
};

template <AnyVector SrcT, AnyVector DstT, int Select> struct SumSqr1Or2
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        if constexpr (Select == 1)
        {
            DstT temp(aSrc1);
            temp.Sqr();
            aDst += temp;
        }
        else
        {
            DstT temp(aSrc2);
            temp.Sqr();
            aDst += temp;
        }
    }
};

template <AnyVector SrcT, AnyVector DstT> struct NormInf
{
    DEVICE_CODE void operator()(const SrcT &aSrc, DstT &aDst) const
    {
        DstT temp(aSrc);
        temp.Abs();
        aDst.Max(temp);
    }
    // special case needed to compute NormInf on Src2 for NormRelInf
    DEVICE_CODE void operator()(const SrcT & /*aSrc1*/, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT temp(aSrc2);
        temp.Abs();
        aDst.Max(temp);
    }
};

template <AnyVector SrcT, AnyVector DstT> struct NormL1
{
    DEVICE_CODE void operator()(const SrcT &aSrc, DstT &aDst) const
    {
        DstT temp(aSrc);
        temp.Abs();
        aDst += temp;
    }
    // special case needed to compute NormL1 on Src2 for NormRelL1
    DEVICE_CODE void operator()(const SrcT & /*aSrc1*/, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT temp(aSrc2);
        temp.Abs();
        aDst += temp;
    }
};

template <AnyVector SrcT, AnyVector DstT> struct NormL2
{
    DEVICE_CODE void operator()(const SrcT &aSrc, DstT &aDst) const
    {
        DstT temp(aSrc);
        temp.Sqr();
        aDst += temp;
    }
    // special case needed to compute NormL2 on Src2 for NormRelL2
    DEVICE_CODE void operator()(const SrcT & /*aSrc1*/, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT temp(aSrc2);
        temp.Sqr();
        aDst += temp;
    }
};

template <AnyVector SrcT, AnyVector DstT> struct NormRelInf
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT src1(aSrc1);
        DstT src2(aSrc2);
        DstT temp = src1 - src2;
        temp.Abs();
        aDst.Max(temp);
    }
};

template <AnyVector SrcT, AnyVector DstT> struct NormRelL1
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT src1(aSrc1);
        DstT src2(aSrc2);
        DstT temp = src1 - src2;
        temp.Abs();
        aDst += temp;
    }
};

template <AnyVector SrcT, AnyVector DstT> struct NormRelL2
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT src1(aSrc1);
        DstT src2(aSrc2);
        DstT temp = src1 - src2;
        temp.Sqr();
        aDst += temp;
    }
};

template <AnyVector SrcT, AnyVector DstT> struct NormDiffInf
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT src1(aSrc1);
        DstT src2(aSrc2);
        DstT temp = src1 - src2;
        temp.Abs();
        aDst.Max(temp);
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires ComplexVector<SrcT>
    {
        SrcT temp = aSrc1 - aSrc2;
        DstT temp2(temp.Magnitude());
        aDst.Max(temp2);
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires ComplexVector<SrcT> && RealOrComplexIntVector<SrcT>
    {
        // avoid clamping for integer differences by converting to float, caution for c_int64 though!
        same_vector_size_different_type_t<SrcT, c_float> temp =
            same_vector_size_different_type_t<SrcT, c_float>(aSrc1) -
            same_vector_size_different_type_t<SrcT, c_float>(aSrc2);
        DstT temp2(temp.Magnitude());
        aDst.Max(temp2);
    }
};

template <AnyVector SrcT, AnyVector DstT> struct NormDiffL1
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT src1(aSrc1);
        DstT src2(aSrc2);
        DstT temp = src1 - src2;
        temp.Abs();
        aDst += temp;
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires ComplexVector<SrcT>
    {
        SrcT temp = aSrc1 - aSrc2;
        DstT temp2(temp.Magnitude());
        aDst += temp2;
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires ComplexVector<SrcT> && RealOrComplexIntVector<SrcT>
    {
        // avoid clamping for integer differences by converting to float, caution for c_int64 though!
        same_vector_size_different_type_t<SrcT, c_float> temp =
            same_vector_size_different_type_t<SrcT, c_float>(aSrc1) -
            same_vector_size_different_type_t<SrcT, c_float>(aSrc2);
        DstT temp2(temp.Magnitude());
        aDst += temp2;
    }
};

template <AnyVector SrcT, AnyVector DstT> struct NormDiffL2
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT src1(aSrc1);
        DstT src2(aSrc2);
        DstT temp = src1 - src2;
        temp.Sqr();
        aDst += temp;
    }
};

template <AnyVector SrcT, AnyVector DstT> struct DotProduct
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT src1(aSrc1);
        DstT src2(aSrc2);
        DstT temp = src1 * src2;
        aDst += temp;
    }
};

template <AnyVector SrcT, AnyVector DstT> struct AverageRelativeError
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT src1(aSrc1);
        DstT src2(aSrc2);
        DstT temp = src1 - src2;
        temp.Abs();
        src1.Abs();
        src2.Abs();
        DstT srcMax = DstT::Max(src1, src2);

        if (srcMax.x != static_cast<remove_vector_t<DstT>>(0))
        {
            temp.x /= srcMax.x;
        }
        else
        {
            temp.x = static_cast<remove_vector_t<DstT>>(0);
        }

        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            if (srcMax.y != static_cast<remove_vector_t<DstT>>(0))
            {
                temp.y /= srcMax.y;
            }
            else
            {
                temp.y = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            if (srcMax.z != static_cast<remove_vector_t<DstT>>(0))
            {
                temp.z /= srcMax.z;
            }
            else
            {
                temp.z = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            if (srcMax.w != static_cast<remove_vector_t<DstT>>(0))
            {
                temp.w /= srcMax.w;
            }
            else
            {
                temp.w = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        aDst += temp;
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires ComplexVector<SrcT>
    {
        DstT src1(aSrc1.Magnitude());
        DstT src2(aSrc2.Magnitude());
        DstT srcMax = DstT::Max(src1, src2);
        SrcT temp   = aSrc1 - aSrc2;
        DstT temp2(temp.Magnitude());

        if (srcMax.x != static_cast<remove_vector_t<DstT>>(0))
        {
            temp2.x /= srcMax.x;
        }
        else
        {
            temp2.x = static_cast<remove_vector_t<DstT>>(0);
        }

        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            if (srcMax.y != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.y /= srcMax.y;
            }
            else
            {
                temp2.y = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            if (srcMax.z != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.z /= srcMax.z;
            }
            else
            {
                temp2.z = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            if (srcMax.w != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.w /= srcMax.w;
            }
            else
            {
                temp2.w = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        aDst += temp2;
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires ComplexVector<SrcT> && RealOrComplexIntVector<SrcT>
    {
        same_vector_size_different_type_t<SrcT, c_float> tempSrc1(aSrc1);
        same_vector_size_different_type_t<SrcT, c_float> tempSrc2(aSrc2);

        DstT src1(tempSrc1.Magnitude());
        DstT src2(tempSrc2.Magnitude());
        DstT srcMax                                           = DstT::Max(src1, src2);
        same_vector_size_different_type_t<SrcT, c_float> temp = tempSrc1 - tempSrc2;
        DstT temp2(temp.Magnitude());

        if (srcMax.x != static_cast<remove_vector_t<DstT>>(0))
        {
            temp2.x /= srcMax.x;
        }
        else
        {
            temp2.x = static_cast<remove_vector_t<DstT>>(0);
        }

        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            if (srcMax.y != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.y /= srcMax.y;
            }
            else
            {
                temp2.y = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            if (srcMax.z != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.z /= srcMax.z;
            }
            else
            {
                temp2.z = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            if (srcMax.w != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.w /= srcMax.w;
            }
            else
            {
                temp2.w = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        aDst += temp2;
    }
};

template <AnyVector SrcT, AnyVector DstT> struct MaximumRelativeError
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
    {
        DstT src1(aSrc1);
        DstT src2(aSrc2);
        DstT temp = src1 - src2;
        temp.Abs();
        src1.Abs();
        src2.Abs();
        DstT srcMax = DstT::Max(src1, src2);

        if (srcMax.x != static_cast<remove_vector_t<DstT>>(0))
        {
            temp.x /= srcMax.x;
        }
        else
        {
            temp.x = static_cast<remove_vector_t<DstT>>(0);
        }

        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            if (srcMax.y != static_cast<remove_vector_t<DstT>>(0))
            {
                temp.y /= srcMax.y;
            }
            else
            {
                temp.y = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            if (srcMax.z != static_cast<remove_vector_t<DstT>>(0))
            {
                temp.z /= srcMax.z;
            }
            else
            {
                temp.z = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            if (srcMax.w != static_cast<remove_vector_t<DstT>>(0))
            {
                temp.w /= srcMax.w;
            }
            else
            {
                temp.w = static_cast<remove_vector_t<DstT>>(0);
            }
        }

        aDst.Max(temp);
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires ComplexVector<SrcT>
    {
        DstT src1(aSrc1.Magnitude());
        DstT src2(aSrc2.Magnitude());
        DstT srcMax = DstT::Max(src1, src2);
        SrcT temp   = aSrc1 - aSrc2;
        DstT temp2(temp.Magnitude());

        if (srcMax.x != static_cast<remove_vector_t<DstT>>(0))
        {
            temp2.x /= srcMax.x;
        }
        else
        {
            temp2.x = static_cast<remove_vector_t<DstT>>(0);
        }

        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            if (srcMax.y != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.y /= srcMax.y;
            }
            else
            {
                temp2.y = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            if (srcMax.z != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.z /= srcMax.z;
            }
            else
            {
                temp2.z = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            if (srcMax.w != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.w /= srcMax.w;
            }
            else
            {
                temp2.w = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        aDst.Max(temp2);
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires ComplexVector<SrcT> && RealOrComplexIntVector<SrcT>
    {
        same_vector_size_different_type_t<SrcT, c_float> tempSrc1(aSrc1);
        same_vector_size_different_type_t<SrcT, c_float> tempSrc2(aSrc2);
        DstT src1(tempSrc1.Magnitude());
        DstT src2(tempSrc2.Magnitude());
        DstT srcMax                                           = DstT::Max(src1, src2);
        same_vector_size_different_type_t<SrcT, c_float> temp = tempSrc1 - tempSrc2;
        DstT temp2(temp.Magnitude());

        if (srcMax.x != static_cast<remove_vector_t<DstT>>(0))
        {
            temp2.x /= srcMax.x;
        }
        else
        {
            temp2.x = static_cast<remove_vector_t<DstT>>(0);
        }

        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            if (srcMax.y != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.y /= srcMax.y;
            }
            else
            {
                temp2.y = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            if (srcMax.z != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.z /= srcMax.z;
            }
            else
            {
                temp2.z = static_cast<remove_vector_t<DstT>>(0);
            }
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            if (srcMax.w != static_cast<remove_vector_t<DstT>>(0))
            {
                temp2.w /= srcMax.w;
            }
            else
            {
                temp2.w = static_cast<remove_vector_t<DstT>>(0);
            }
        }

        aDst.Max(temp2);
    }
};

} // namespace mpp
