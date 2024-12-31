#pragma once
#include <common/defines.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp
{
template <AnyVector T> struct AbsDiff
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = T::AbsDiff(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst.AbsDiff(aSrc1);
    }
};

template <AnyVector T> struct Add
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = aSrc1 + aSrc2;
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst += aSrc1;
    }
};

template <AnyVector T> struct Sub
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = aSrc1 - aSrc2;
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst -= aSrc1;
    }
};

/// <summary>
/// Inverted argument order for inplace sub: aSrcDst = aSrc1 - aSrcDst
/// </summary>
template <AnyVector T> struct SubInv
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst.SubInv(aSrc1);
    }
};

template <AnyVector T> struct Mul
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = aSrc1 * aSrc2;
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst *= aSrc1;
    }
};

template <AnyVector T> struct Div
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = aSrc1 / aSrc2;
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst /= aSrc1;
    }
};

/// <summary>
/// Inverted argument order for inplace div: aSrcDst = aSrc1 / aSrcDst
/// </summary>
template <AnyVector T> struct DivInv
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst.DivInv(aSrc1);
    }
};

template <RealIntVector T> struct And
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = T::And(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst.And(aSrc1);
    }
};

template <RealIntVector T> struct Or
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = T::Or(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst.Or(aSrc1);
    }
};

template <RealIntVector T> struct Xor
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = T::Xor(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst.Xor(aSrc1);
    }
};

template <RealIntVector T> struct LShift
{
    const int Shift;

    LShift(int aShift) : Shift(aShift)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Lshift(aSrc1, Shift);
    }
    DEVICE_CODE void operator()(int aSrc1, T &aSrcDst)
    {
        aSrcDst.LShift(Shift);
    }
};

template <RealIntVector T> struct RShift
{
    const int Shift;

    RShift(int aShift) : Shift(aShift)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Rshift(aSrc1, Shift);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Rshift(Shift);
    }
};

template <AnyVector T> struct AddSqr
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst += T::Sqr(aSrc1);
    }
};

template <AnyVector T> struct Min
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = T::Min(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst.Min(aSrc1);
    }
};

template <AnyVector T> struct Max
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, T &aDst)
    {
        aDst = T::Max(aSrc1, aSrc2);
    }
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst.Max(aSrc1);
    }
};

template <AnyVector T> struct Eq
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, bool &aDst)
    {
        aDst = aSrc1 == aSrc2;
    }
};

template <AnyVector T> struct Ge
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, bool &aDst)
    {
        aDst = aSrc1 >= aSrc2;
    }
};

template <RealVector T> struct Gt
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, bool &aDst)
    {
        aDst = aSrc1 > aSrc2;
    }
};

template <RealVector T> struct Le
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, bool &aDst)
    {
        aDst = aSrc1 <= aSrc2;
    }
};

template <RealVector T> struct Lt
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, bool &aDst)
    {
        aDst = aSrc1 < aSrc2;
    }
};

template <AnyVector T> struct NEq
{
    DEVICE_CODE void operator()(const T &aSrc1, const T &aSrc2, bool &aDst)
    {
        aDst = aSrc1 != aSrc2;
    }
};
} // namespace opp
