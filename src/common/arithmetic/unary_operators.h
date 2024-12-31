#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp
{
template <AnyVector T> struct Set
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = aSrc1;
    }
};

template <AnyVector T> struct Neg
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = -aSrc1;
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst = -aSrcDst;
    }
};

template <RealIntVector T> struct Not
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Not(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Not();
    }
};

template <RealVector T> struct Sqr
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Sqr(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Sqr();
    }
};

template <RealVector T> struct Sqrt
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Sqrt(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Sqrt();
    }
};

template <RealVector T> struct Ln
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Ln(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Ln();
    }
};

template <RealVector T> struct Exp
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Exp(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Exp();
    }
};

template <RealVector T> struct Abs
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Abs(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Abs();
    }
};

template <RealOrComplexFloatingVector T> struct Round
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Round(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Round();
    }
};

template <RealOrComplexFloatingVector T> struct Floor
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Floor(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Floor();
    }
};

template <RealOrComplexFloatingVector T> struct Ceil
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Ceil(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Ceil();
    }
};

template <RealOrComplexFloatingVector T> struct RoundNearest
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::RoundNearest(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.RoundNearest();
    }
};

template <RealOrComplexFloatingVector T> struct RoundZero
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::RoundZero(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.RoundZero();
    }
};

template <ComplexVector T> struct Magnitude
{
    DEVICE_CODE void operator()(const T &aSrc1, complex_basetype_t<T> &aDst)
    {
        aDst = aSrc1.Magnitude();
    }
};

template <ComplexVector T> struct MagnitudeSqr
{
    DEVICE_CODE void operator()(const T &aSrc1, complex_basetype_t<T> &aDst)
    {
        aDst = aSrc1.MagnitudeSqr();
    }
};
} // namespace opp
