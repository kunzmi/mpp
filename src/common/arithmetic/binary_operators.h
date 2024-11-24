#pragma once
#include <common/defines.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp
{
template <VectorType T> struct AbsDiff
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

template <VectorOrComplexType T> struct Add
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

template <VectorOrComplexType T> struct Sub
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
template <VectorOrComplexType T> struct SubInv
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst = aSrc1 - aSrcDst;
    }
};

template <VectorOrComplexType T> struct Mul
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

template <VectorOrComplexType T> struct Div
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
template <VectorOrComplexType T> struct DivInv
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst = aSrc1 / aSrcDst;
    }
};

template <IntVectorType T> struct And
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

template <IntVectorType T> struct Or
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

template <IntVectorType T> struct Xor
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

template <IntVectorType T> struct LShift
{
    int Shift;

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

template <IntVectorType T> struct RShift
{
    int Shift;

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

template <VectorType T> struct AddSqr
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aSrcDst)
    {
        aSrcDst += T::Sqr(aSrc1);
    }
};
} // namespace opp
