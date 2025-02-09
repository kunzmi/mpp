#include "bfloat16.h"     //NOLINT(misc-include-cleaner)
#include "complex.h"      //NOLINT
#include "defines.h"      //NOLINT(misc-include-cleaner)
#include "half_fp16.h"    //NOLINT(misc-include-cleaner)
#include "vector1.h"      //NOLINT(misc-include-cleaner)
#include "vector1_impl.h" //NOLINT(misc-include-cleaner)
#include <iostream>

namespace opp
{

std::ostream &operator<<(std::ostream &aOs, const Axis1D &aAxis)
{
    switch (aAxis)
    {
        case Axis1D::X:
            aOs << 'X';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X (0).";
    return aOs;
}

std::wostream &operator<<(std::wostream &aOs, const Axis1D &aAxis)
{
    switch (aAxis)
    {
        case Axis1D::X:
            aOs << 'X';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X (0).";
    return aOs;
}

template struct Vector1<sbyte>;
template struct Vector1<byte>;
template struct Vector1<short>;
template struct Vector1<ushort>;
template struct Vector1<int>;
template struct Vector1<uint>;
template struct Vector1<long64>;
template struct Vector1<ulong64>;

template struct Vector1<BFloat16>;
template struct Vector1<HalfFp16>;
template struct Vector1<float>;
template struct Vector1<double>;

template struct Vector1<Complex<sbyte>>;
template struct Vector1<Complex<short>>;
template struct Vector1<Complex<int>>;
template struct Vector1<Complex<BFloat16>>;
template struct Vector1<Complex<HalfFp16>>;
template struct Vector1<Complex<float>>;
template struct Vector1<Complex<double>>;

template Vector1<sbyte>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<short> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<int> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<float> &) noexcept;
template Vector1<sbyte>::Vector1(const Vector1<double> &) noexcept;

template Vector1<byte>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<short> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<int> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<float> &) noexcept;
template Vector1<byte>::Vector1(const Vector1<double> &) noexcept;

template Vector1<short>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<short>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<short>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<short>::Vector1(const Vector1<int> &) noexcept;
template Vector1<short>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<short>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<short>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<short>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<short>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<short>::Vector1(const Vector1<float> &) noexcept;
template Vector1<short>::Vector1(const Vector1<double> &) noexcept;

template Vector1<ushort>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<short> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<int> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<float> &) noexcept;
template Vector1<ushort>::Vector1(const Vector1<double> &) noexcept;

template Vector1<int>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<int>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<int>::Vector1(const Vector1<short> &) noexcept;
template Vector1<int>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<int>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<int>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<int>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<int>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<int>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<int>::Vector1(const Vector1<float> &) noexcept;
template Vector1<int>::Vector1(const Vector1<double> &) noexcept;

template Vector1<uint>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<short> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<int> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<float> &) noexcept;
template Vector1<uint>::Vector1(const Vector1<double> &) noexcept;

template Vector1<long64>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<short> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<int> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<float> &) noexcept;
template Vector1<long64>::Vector1(const Vector1<double> &) noexcept;

template Vector1<ulong64>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<short> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<int> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<float> &) noexcept;
template Vector1<ulong64>::Vector1(const Vector1<double> &) noexcept;

template Vector1<BFloat16>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<short> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<int> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<float> &) noexcept;
template Vector1<BFloat16>::Vector1(const Vector1<double> &) noexcept;

template Vector1<HalfFp16>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<short> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<int> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<float> &) noexcept;
template Vector1<HalfFp16>::Vector1(const Vector1<double> &) noexcept;

template Vector1<float>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<float>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<float>::Vector1(const Vector1<short> &) noexcept;
template Vector1<float>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<float>::Vector1(const Vector1<int> &) noexcept;
template Vector1<float>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<float>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<float>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<float>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<float>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<float>::Vector1(const Vector1<double> &) noexcept;

template Vector1<double>::Vector1(const Vector1<sbyte> &) noexcept;
template Vector1<double>::Vector1(const Vector1<byte> &) noexcept;
template Vector1<double>::Vector1(const Vector1<short> &) noexcept;
template Vector1<double>::Vector1(const Vector1<ushort> &) noexcept;
template Vector1<double>::Vector1(const Vector1<int> &) noexcept;
template Vector1<double>::Vector1(const Vector1<uint> &) noexcept;
template Vector1<double>::Vector1(const Vector1<long64> &) noexcept;
template Vector1<double>::Vector1(const Vector1<ulong64> &) noexcept;
template Vector1<double>::Vector1(const Vector1<BFloat16> &) noexcept;
template Vector1<double>::Vector1(const Vector1<HalfFp16> &) noexcept;
template Vector1<double>::Vector1(const Vector1<float> &) noexcept;

template Vector1<Complex<sbyte>>::Vector1(const Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(const Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(const Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(const Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(const Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(const Vector1<Complex<double>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(const Vector1<sbyte> &) noexcept;

template Vector1<Complex<short>>::Vector1(const Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<short>>::Vector1(const Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<short>>::Vector1(const Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<short>>::Vector1(const Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<short>>::Vector1(const Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<short>>::Vector1(const Vector1<Complex<double>> &) noexcept;
template Vector1<Complex<short>>::Vector1(const Vector1<short> &) noexcept;

template Vector1<Complex<int>>::Vector1(const Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<int>>::Vector1(const Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<int>>::Vector1(const Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<int>>::Vector1(const Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<int>>::Vector1(const Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<int>>::Vector1(const Vector1<Complex<double>> &) noexcept;
template Vector1<Complex<int>>::Vector1(const Vector1<int> &) noexcept;

template Vector1<Complex<BFloat16>>::Vector1(const Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(const Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(const Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(const Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(const Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(const Vector1<Complex<double>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(const Vector1<BFloat16> &) noexcept;

template Vector1<Complex<HalfFp16>>::Vector1(const Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(const Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(const Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(const Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(const Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(const Vector1<Complex<double>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(const Vector1<HalfFp16> &) noexcept;

template Vector1<Complex<float>>::Vector1(const Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<float>>::Vector1(const Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<float>>::Vector1(const Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<float>>::Vector1(const Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<float>>::Vector1(const Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<float>>::Vector1(const Vector1<Complex<double>> &) noexcept;
template Vector1<Complex<float>>::Vector1(const Vector1<float> &) noexcept;

template Vector1<Complex<double>>::Vector1(const Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<double>>::Vector1(const Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<double>>::Vector1(const Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<double>>::Vector1(const Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<double>>::Vector1(const Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<double>>::Vector1(const Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<double>>::Vector1(const Vector1<double> &) noexcept;

template Vector1<sbyte>::Vector1(Vector1<byte> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<short> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<int> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<uint> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<long64> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<float> &) noexcept;
template Vector1<sbyte>::Vector1(Vector1<double> &) noexcept;

template Vector1<byte>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<byte>::Vector1(Vector1<short> &) noexcept;
template Vector1<byte>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<byte>::Vector1(Vector1<int> &) noexcept;
template Vector1<byte>::Vector1(Vector1<uint> &) noexcept;
template Vector1<byte>::Vector1(Vector1<long64> &) noexcept;
template Vector1<byte>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<byte>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<byte>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<byte>::Vector1(Vector1<float> &) noexcept;
template Vector1<byte>::Vector1(Vector1<double> &) noexcept;

template Vector1<short>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<short>::Vector1(Vector1<byte> &) noexcept;
template Vector1<short>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<short>::Vector1(Vector1<int> &) noexcept;
template Vector1<short>::Vector1(Vector1<uint> &) noexcept;
template Vector1<short>::Vector1(Vector1<long64> &) noexcept;
template Vector1<short>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<short>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<short>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<short>::Vector1(Vector1<float> &) noexcept;
template Vector1<short>::Vector1(Vector1<double> &) noexcept;

template Vector1<ushort>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<byte> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<short> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<int> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<uint> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<long64> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<float> &) noexcept;
template Vector1<ushort>::Vector1(Vector1<double> &) noexcept;

template Vector1<int>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<int>::Vector1(Vector1<byte> &) noexcept;
template Vector1<int>::Vector1(Vector1<short> &) noexcept;
template Vector1<int>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<int>::Vector1(Vector1<uint> &) noexcept;
template Vector1<int>::Vector1(Vector1<long64> &) noexcept;
template Vector1<int>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<int>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<int>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<int>::Vector1(Vector1<float> &) noexcept;
template Vector1<int>::Vector1(Vector1<double> &) noexcept;

template Vector1<uint>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<uint>::Vector1(Vector1<byte> &) noexcept;
template Vector1<uint>::Vector1(Vector1<short> &) noexcept;
template Vector1<uint>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<uint>::Vector1(Vector1<int> &) noexcept;
template Vector1<uint>::Vector1(Vector1<long64> &) noexcept;
template Vector1<uint>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<uint>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<uint>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<uint>::Vector1(Vector1<float> &) noexcept;
template Vector1<uint>::Vector1(Vector1<double> &) noexcept;

template Vector1<long64>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<long64>::Vector1(Vector1<byte> &) noexcept;
template Vector1<long64>::Vector1(Vector1<short> &) noexcept;
template Vector1<long64>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<long64>::Vector1(Vector1<int> &) noexcept;
template Vector1<long64>::Vector1(Vector1<uint> &) noexcept;
template Vector1<long64>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<long64>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<long64>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<long64>::Vector1(Vector1<float> &) noexcept;
template Vector1<long64>::Vector1(Vector1<double> &) noexcept;

template Vector1<ulong64>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<byte> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<short> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<int> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<uint> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<long64> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<float> &) noexcept;
template Vector1<ulong64>::Vector1(Vector1<double> &) noexcept;

template Vector1<BFloat16>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<byte> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<short> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<int> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<uint> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<long64> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<float> &) noexcept;
template Vector1<BFloat16>::Vector1(Vector1<double> &) noexcept;

template Vector1<HalfFp16>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<byte> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<short> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<int> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<uint> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<long64> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<float> &) noexcept;
template Vector1<HalfFp16>::Vector1(Vector1<double> &) noexcept;

template Vector1<float>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<float>::Vector1(Vector1<byte> &) noexcept;
template Vector1<float>::Vector1(Vector1<short> &) noexcept;
template Vector1<float>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<float>::Vector1(Vector1<int> &) noexcept;
template Vector1<float>::Vector1(Vector1<uint> &) noexcept;
template Vector1<float>::Vector1(Vector1<long64> &) noexcept;
template Vector1<float>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<float>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<float>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<float>::Vector1(Vector1<double> &) noexcept;

template Vector1<double>::Vector1(Vector1<sbyte> &) noexcept;
template Vector1<double>::Vector1(Vector1<byte> &) noexcept;
template Vector1<double>::Vector1(Vector1<short> &) noexcept;
template Vector1<double>::Vector1(Vector1<ushort> &) noexcept;
template Vector1<double>::Vector1(Vector1<int> &) noexcept;
template Vector1<double>::Vector1(Vector1<uint> &) noexcept;
template Vector1<double>::Vector1(Vector1<long64> &) noexcept;
template Vector1<double>::Vector1(Vector1<ulong64> &) noexcept;
template Vector1<double>::Vector1(Vector1<BFloat16> &) noexcept;
template Vector1<double>::Vector1(Vector1<HalfFp16> &) noexcept;
template Vector1<double>::Vector1(Vector1<float> &) noexcept;

template Vector1<Complex<sbyte>>::Vector1(Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<sbyte>>::Vector1(Vector1<Complex<double>> &) noexcept;

template Vector1<Complex<short>>::Vector1(Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<short>>::Vector1(Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<short>>::Vector1(Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<short>>::Vector1(Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<short>>::Vector1(Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<short>>::Vector1(Vector1<Complex<double>> &) noexcept;

template Vector1<Complex<int>>::Vector1(Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<int>>::Vector1(Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<int>>::Vector1(Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<int>>::Vector1(Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<int>>::Vector1(Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<int>>::Vector1(Vector1<Complex<double>> &) noexcept;

template Vector1<Complex<BFloat16>>::Vector1(Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<BFloat16>>::Vector1(Vector1<Complex<double>> &) noexcept;

template Vector1<Complex<HalfFp16>>::Vector1(Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(Vector1<Complex<float>> &) noexcept;
template Vector1<Complex<HalfFp16>>::Vector1(Vector1<Complex<double>> &) noexcept;

template Vector1<Complex<float>>::Vector1(Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<float>>::Vector1(Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<float>>::Vector1(Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<float>>::Vector1(Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<float>>::Vector1(Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<float>>::Vector1(Vector1<Complex<double>> &) noexcept;

template Vector1<Complex<double>>::Vector1(Vector1<Complex<sbyte>> &) noexcept;
template Vector1<Complex<double>>::Vector1(Vector1<Complex<short>> &) noexcept;
template Vector1<Complex<double>>::Vector1(Vector1<Complex<int>> &) noexcept;
template Vector1<Complex<double>>::Vector1(Vector1<Complex<BFloat16>> &) noexcept;
template Vector1<Complex<double>>::Vector1(Vector1<Complex<HalfFp16>> &) noexcept;
template Vector1<Complex<double>>::Vector1(Vector1<Complex<float>> &) noexcept;

template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<sbyte>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<byte>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<short>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<ushort>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<int>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<uint>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<long64>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<ulong64>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<float>() noexcept;
template Vector1<sbyte> &Vector1<sbyte>::ClampToTargetType<double>() noexcept;

template Vector1<byte> &Vector1<byte>::ClampToTargetType<sbyte>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<byte>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<short>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<ushort>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<int>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<uint>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<long64>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<ulong64>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<float>() noexcept;
template Vector1<byte> &Vector1<byte>::ClampToTargetType<double>() noexcept;

template Vector1<short> &Vector1<short>::ClampToTargetType<byte>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<sbyte>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<short>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<ushort>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<int>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<uint>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<long64>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<ulong64>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<float>() noexcept;
template Vector1<short> &Vector1<short>::ClampToTargetType<double>() noexcept;

template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<byte>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<sbyte>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<short>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<ushort>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<int>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<uint>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<long64>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<ulong64>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<float>() noexcept;
template Vector1<ushort> &Vector1<ushort>::ClampToTargetType<double>() noexcept;

template Vector1<int> &Vector1<int>::ClampToTargetType<byte>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<sbyte>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<short>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<ushort>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<int>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<uint>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<long64>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<ulong64>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<float>() noexcept;
template Vector1<int> &Vector1<int>::ClampToTargetType<double>() noexcept;

template Vector1<uint> &Vector1<uint>::ClampToTargetType<byte>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<sbyte>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<short>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<ushort>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<int>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<uint>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<long64>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<ulong64>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<float>() noexcept;
template Vector1<uint> &Vector1<uint>::ClampToTargetType<double>() noexcept;

template Vector1<long64> &Vector1<long64>::ClampToTargetType<byte>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<sbyte>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<short>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<ushort>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<int>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<uint>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<long64>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<ulong64>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<float>() noexcept;
template Vector1<long64> &Vector1<long64>::ClampToTargetType<double>() noexcept;

template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<byte>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<sbyte>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<short>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<ushort>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<int>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<uint>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<long64>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<ulong64>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<float>() noexcept;
template Vector1<ulong64> &Vector1<ulong64>::ClampToTargetType<double>() noexcept;

template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<byte>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<sbyte>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<short>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<ushort>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<int>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<uint>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<long64>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<ulong64>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<float>() noexcept;
template Vector1<BFloat16> &Vector1<BFloat16>::ClampToTargetType<double>() noexcept;

template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<byte>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<sbyte>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<short>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<ushort>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<int>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<uint>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<long64>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<ulong64>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<float>() noexcept;
template Vector1<HalfFp16> &Vector1<HalfFp16>::ClampToTargetType<double>() noexcept;

template Vector1<float> &Vector1<float>::ClampToTargetType<byte>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<sbyte>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<short>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<ushort>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<int>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<uint>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<long64>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<ulong64>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<float>() noexcept;
template Vector1<float> &Vector1<float>::ClampToTargetType<double>() noexcept;

template Vector1<double> &Vector1<double>::ClampToTargetType<byte>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<sbyte>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<short>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<ushort>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<int>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<uint>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<long64>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<ulong64>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<float>() noexcept;
template Vector1<double> &Vector1<double>::ClampToTargetType<double>() noexcept;

template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<sbyte>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<byte>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<short>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<ushort>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<int>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<uint>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<long64>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<ulong64>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<float>() noexcept;
template Vector1<Complex<sbyte>> &Vector1<Complex<sbyte>>::ClampToTargetType<double>() noexcept;

template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<sbyte>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<byte>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<short>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<ushort>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<int>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<uint>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<long64>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<ulong64>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<float>() noexcept;
template Vector1<Complex<short>> &Vector1<Complex<short>>::ClampToTargetType<double>() noexcept;

template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<sbyte>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<byte>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<short>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<ushort>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<int>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<uint>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<long64>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<ulong64>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<float>() noexcept;
template Vector1<Complex<int>> &Vector1<Complex<int>>::ClampToTargetType<double>() noexcept;

template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<sbyte>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<byte>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<short>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<ushort>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<int>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<uint>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<long64>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<ulong64>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<float>() noexcept;
template Vector1<Complex<long64>> &Vector1<Complex<long64>>::ClampToTargetType<double>() noexcept;

template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<sbyte>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<byte>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<short>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<ushort>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<int>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<uint>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<long64>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<ulong64>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<float>() noexcept;
template Vector1<Complex<BFloat16>> &Vector1<Complex<BFloat16>>::ClampToTargetType<double>() noexcept;

template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<sbyte>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<byte>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<short>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<ushort>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<int>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<uint>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<long64>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<ulong64>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<float>() noexcept;
template Vector1<Complex<HalfFp16>> &Vector1<Complex<HalfFp16>>::ClampToTargetType<double>() noexcept;

template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<sbyte>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<byte>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<short>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<ushort>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<int>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<uint>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<long64>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<ulong64>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<float>() noexcept;
template Vector1<Complex<float>> &Vector1<Complex<float>>::ClampToTargetType<double>() noexcept;

template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<sbyte>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<byte>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<short>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<ushort>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<int>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<uint>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<long64>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<ulong64>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<BFloat16>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<float>() noexcept;
template Vector1<Complex<double>> &Vector1<Complex<double>>::ClampToTargetType<double>() noexcept;

template std::ostream &operator<<(std::ostream &aOs, const Vector1<sbyte> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<sbyte> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<sbyte> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<sbyte> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<byte> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<byte> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<byte> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<byte> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<short> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<short> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<short> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<short> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<ushort> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<ushort> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<ushort> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<ushort> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<int> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<int> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<int> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<int> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<uint> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<uint> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<uint> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<uint> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<long64> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<long64> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<long64> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<long64> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<ulong64> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<ulong64> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<ulong64> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<ulong64> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<BFloat16> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<BFloat16> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<BFloat16> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<BFloat16> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<HalfFp16> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<HalfFp16> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<HalfFp16> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<HalfFp16> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<float> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<float> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<float> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<float> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<double> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<double> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<double> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<double> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<Complex<sbyte>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<Complex<sbyte>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<Complex<sbyte>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<Complex<sbyte>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<Complex<short>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<Complex<short>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<Complex<short>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<Complex<short>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<Complex<int>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<Complex<int>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<Complex<int>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<Complex<int>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<Complex<BFloat16>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<Complex<BFloat16>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<Complex<BFloat16>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<Complex<BFloat16>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<Complex<HalfFp16>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<Complex<HalfFp16>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<Complex<HalfFp16>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<Complex<HalfFp16>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<Complex<float>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<Complex<float>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<Complex<float>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<Complex<float>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector1<Complex<double>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector1<Complex<double>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector1<Complex<double>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector1<Complex<double>> &aVec);
} // namespace opp