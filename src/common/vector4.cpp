#include "bfloat16.h"     //NOLINT(misc-include-cleaner)
#include "complex.h"      //NOLINT
#include "defines.h"      //NOLINT(misc-include-cleaner)
#include "half_fp16.h"    //NOLINT(misc-include-cleaner)
#include "vector4.h"      //NOLINT(misc-include-cleaner)
#include "vector4_impl.h" //NOLINT(misc-include-cleaner)
#include <iostream>

namespace mpp
{

std::ostream &operator<<(std::ostream &aOs, const Axis4D &aAxis)
{
    switch (aAxis)
    {
        case Axis4D::X:
            aOs << 'X';
            return aOs;
        case Axis4D::Y:
            aOs << 'Y';
            return aOs;
        case Axis4D::Z:
            aOs << 'Z';
            return aOs;
        case Axis4D::W:
            aOs << 'W';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X, Y, Z or W (0, 1, 2 or 3).";
    return aOs;
}

std::wostream &operator<<(std::wostream &aOs, const Axis4D &aAxis)
{
    switch (aAxis)
    {
        case Axis4D::X:
            aOs << 'X';
            return aOs;
        case Axis4D::Y:
            aOs << 'Y';
            return aOs;
        case Axis4D::Z:
            aOs << 'Z';
            return aOs;
        case Axis4D::W:
            aOs << 'W';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X, Y, Z or W (0, 1, 2 or 3).";
    return aOs;
}

template struct Vector4<sbyte>;
template struct Vector4<byte>;
template struct Vector4<short>;
template struct Vector4<ushort>;
template struct Vector4<int>;
template struct Vector4<uint>;
template struct Vector4<long64>;
template struct Vector4<ulong64>;

template struct Vector4<BFloat16>;
template struct Vector4<HalfFp16>;
template struct Vector4<float>;
template struct Vector4<double>;

template struct Vector4<Complex<sbyte>>;
template struct Vector4<Complex<short>>;
template struct Vector4<Complex<int>>;
template struct Vector4<Complex<long64>>;
template struct Vector4<Complex<BFloat16>>;
template struct Vector4<Complex<HalfFp16>>;
template struct Vector4<Complex<float>>;
template struct Vector4<Complex<double>>;

template Vector4<sbyte>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<short> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<int> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<float> &) noexcept;
template Vector4<sbyte>::Vector4(const Vector4<double> &) noexcept;

template Vector4<byte>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<short> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<int> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<float> &) noexcept;
template Vector4<byte>::Vector4(const Vector4<double> &) noexcept;

template Vector4<short>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<short>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<short>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<short>::Vector4(const Vector4<int> &) noexcept;
template Vector4<short>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<short>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<short>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<short>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<short>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<short>::Vector4(const Vector4<float> &) noexcept;
template Vector4<short>::Vector4(const Vector4<double> &) noexcept;

template Vector4<ushort>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<short> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<int> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<float> &) noexcept;
template Vector4<ushort>::Vector4(const Vector4<double> &) noexcept;

template Vector4<int>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<int>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<int>::Vector4(const Vector4<short> &) noexcept;
template Vector4<int>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<int>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<int>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<int>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<int>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<int>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<int>::Vector4(const Vector4<float> &) noexcept;
template Vector4<int>::Vector4(const Vector4<double> &) noexcept;

template Vector4<uint>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<short> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<int> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<float> &) noexcept;
template Vector4<uint>::Vector4(const Vector4<double> &) noexcept;

template Vector4<long64>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<short> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<int> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<float> &) noexcept;
template Vector4<long64>::Vector4(const Vector4<double> &) noexcept;

template Vector4<ulong64>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<short> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<int> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<float> &) noexcept;
template Vector4<ulong64>::Vector4(const Vector4<double> &) noexcept;

template Vector4<BFloat16>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<short> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<int> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<float> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<double> &) noexcept;
template Vector4<BFloat16>::Vector4(const Vector4<float> &, RoundingMode);

template Vector4<HalfFp16>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<short> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<int> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<float> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<double> &) noexcept;
template Vector4<HalfFp16>::Vector4(const Vector4<float> &, RoundingMode);

template Vector4<float>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<float>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<float>::Vector4(const Vector4<short> &) noexcept;
template Vector4<float>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<float>::Vector4(const Vector4<int> &) noexcept;
template Vector4<float>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<float>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<float>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<float>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<float>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<float>::Vector4(const Vector4<double> &) noexcept;

template Vector4<double>::Vector4(const Vector4<sbyte> &) noexcept;
template Vector4<double>::Vector4(const Vector4<byte> &) noexcept;
template Vector4<double>::Vector4(const Vector4<short> &) noexcept;
template Vector4<double>::Vector4(const Vector4<ushort> &) noexcept;
template Vector4<double>::Vector4(const Vector4<int> &) noexcept;
template Vector4<double>::Vector4(const Vector4<uint> &) noexcept;
template Vector4<double>::Vector4(const Vector4<long64> &) noexcept;
template Vector4<double>::Vector4(const Vector4<ulong64> &) noexcept;
template Vector4<double>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<double>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<double>::Vector4(const Vector4<float> &) noexcept;

template Vector4<Complex<sbyte>>::Vector4(const Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(const Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(const Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(const Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(const Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(const Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(const Vector4<Complex<double>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(const Vector4<sbyte> &) noexcept;

template Vector4<Complex<short>>::Vector4(const Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<short>>::Vector4(const Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<short>>::Vector4(const Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<short>>::Vector4(const Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<short>>::Vector4(const Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<short>>::Vector4(const Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<short>>::Vector4(const Vector4<Complex<double>> &) noexcept;
template Vector4<Complex<short>>::Vector4(const Vector4<short> &) noexcept;

template Vector4<Complex<int>>::Vector4(const Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<int>>::Vector4(const Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<int>>::Vector4(const Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<int>>::Vector4(const Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<int>>::Vector4(const Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<int>>::Vector4(const Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<int>>::Vector4(const Vector4<Complex<double>> &) noexcept;
template Vector4<Complex<int>>::Vector4(const Vector4<int> &) noexcept;

template Vector4<Complex<long64>>::Vector4(const Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(const Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(const Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(const Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(const Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(const Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(const Vector4<Complex<double>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(const Vector4<long64> &) noexcept;

template Vector4<Complex<BFloat16>>::Vector4(const Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(const Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(const Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(const Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(const Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(const Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(const Vector4<Complex<double>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(const Vector4<BFloat16> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(const Vector4<Complex<float>> &, RoundingMode);

template Vector4<Complex<HalfFp16>>::Vector4(const Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(const Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(const Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(const Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(const Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(const Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(const Vector4<Complex<double>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(const Vector4<HalfFp16> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(const Vector4<Complex<float>> &, RoundingMode);

template Vector4<Complex<float>>::Vector4(const Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<float>>::Vector4(const Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<float>>::Vector4(const Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<float>>::Vector4(const Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<float>>::Vector4(const Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<float>>::Vector4(const Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<float>>::Vector4(const Vector4<Complex<double>> &) noexcept;
template Vector4<Complex<float>>::Vector4(const Vector4<float> &) noexcept;

template Vector4<Complex<double>>::Vector4(const Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<double>>::Vector4(const Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<double>>::Vector4(const Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<double>>::Vector4(const Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<double>>::Vector4(const Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<double>>::Vector4(const Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<double>>::Vector4(const Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<double>>::Vector4(const Vector4<double> &) noexcept;

template Vector4<sbyte>::Vector4(Vector4<byte> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<short> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<int> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<uint> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<long64> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<float> &) noexcept;
template Vector4<sbyte>::Vector4(Vector4<double> &) noexcept;

template Vector4<byte>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<byte>::Vector4(Vector4<short> &) noexcept;
template Vector4<byte>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<byte>::Vector4(Vector4<int> &) noexcept;
template Vector4<byte>::Vector4(Vector4<uint> &) noexcept;
template Vector4<byte>::Vector4(Vector4<long64> &) noexcept;
template Vector4<byte>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<byte>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<byte>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<byte>::Vector4(Vector4<float> &) noexcept;
template Vector4<byte>::Vector4(Vector4<double> &) noexcept;

template Vector4<short>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<short>::Vector4(Vector4<byte> &) noexcept;
template Vector4<short>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<short>::Vector4(Vector4<int> &) noexcept;
template Vector4<short>::Vector4(Vector4<uint> &) noexcept;
template Vector4<short>::Vector4(Vector4<long64> &) noexcept;
template Vector4<short>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<short>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<short>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<short>::Vector4(Vector4<float> &) noexcept;
template Vector4<short>::Vector4(Vector4<double> &) noexcept;

template Vector4<ushort>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<byte> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<short> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<int> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<uint> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<long64> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<float> &) noexcept;
template Vector4<ushort>::Vector4(Vector4<double> &) noexcept;

template Vector4<int>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<int>::Vector4(Vector4<byte> &) noexcept;
template Vector4<int>::Vector4(Vector4<short> &) noexcept;
template Vector4<int>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<int>::Vector4(Vector4<uint> &) noexcept;
template Vector4<int>::Vector4(Vector4<long64> &) noexcept;
template Vector4<int>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<int>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<int>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<int>::Vector4(Vector4<float> &) noexcept;
template Vector4<int>::Vector4(Vector4<double> &) noexcept;

template Vector4<uint>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<uint>::Vector4(Vector4<byte> &) noexcept;
template Vector4<uint>::Vector4(Vector4<short> &) noexcept;
template Vector4<uint>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<uint>::Vector4(Vector4<int> &) noexcept;
template Vector4<uint>::Vector4(Vector4<long64> &) noexcept;
template Vector4<uint>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<uint>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<uint>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<uint>::Vector4(Vector4<float> &) noexcept;
template Vector4<uint>::Vector4(Vector4<double> &) noexcept;

template Vector4<long64>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<long64>::Vector4(Vector4<byte> &) noexcept;
template Vector4<long64>::Vector4(Vector4<short> &) noexcept;
template Vector4<long64>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<long64>::Vector4(Vector4<int> &) noexcept;
template Vector4<long64>::Vector4(Vector4<uint> &) noexcept;
template Vector4<long64>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<long64>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<long64>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<long64>::Vector4(Vector4<float> &) noexcept;
template Vector4<long64>::Vector4(Vector4<double> &) noexcept;

template Vector4<ulong64>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<byte> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<short> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<int> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<uint> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<long64> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<float> &) noexcept;
template Vector4<ulong64>::Vector4(Vector4<double> &) noexcept;

template Vector4<BFloat16>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<byte> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<short> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<int> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<uint> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<long64> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<float> &) noexcept;
template Vector4<BFloat16>::Vector4(Vector4<double> &) noexcept;

template Vector4<HalfFp16>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<byte> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<short> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<int> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<uint> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<long64> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<float> &) noexcept;
template Vector4<HalfFp16>::Vector4(Vector4<double> &) noexcept;

template Vector4<float>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<float>::Vector4(Vector4<byte> &) noexcept;
template Vector4<float>::Vector4(Vector4<short> &) noexcept;
template Vector4<float>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<float>::Vector4(Vector4<int> &) noexcept;
template Vector4<float>::Vector4(Vector4<uint> &) noexcept;
template Vector4<float>::Vector4(Vector4<long64> &) noexcept;
template Vector4<float>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<float>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<float>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<float>::Vector4(Vector4<double> &) noexcept;

template Vector4<double>::Vector4(Vector4<sbyte> &) noexcept;
template Vector4<double>::Vector4(Vector4<byte> &) noexcept;
template Vector4<double>::Vector4(Vector4<short> &) noexcept;
template Vector4<double>::Vector4(Vector4<ushort> &) noexcept;
template Vector4<double>::Vector4(Vector4<int> &) noexcept;
template Vector4<double>::Vector4(Vector4<uint> &) noexcept;
template Vector4<double>::Vector4(Vector4<long64> &) noexcept;
template Vector4<double>::Vector4(Vector4<ulong64> &) noexcept;
template Vector4<double>::Vector4(Vector4<BFloat16> &) noexcept;
template Vector4<double>::Vector4(Vector4<HalfFp16> &) noexcept;
template Vector4<double>::Vector4(Vector4<float> &) noexcept;

template Vector4<Complex<sbyte>>::Vector4(Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<sbyte>>::Vector4(Vector4<Complex<double>> &) noexcept;

template Vector4<Complex<short>>::Vector4(Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<short>>::Vector4(Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<short>>::Vector4(Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<short>>::Vector4(Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<short>>::Vector4(Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<short>>::Vector4(Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<short>>::Vector4(Vector4<Complex<double>> &) noexcept;

template Vector4<Complex<int>>::Vector4(Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<int>>::Vector4(Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<int>>::Vector4(Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<int>>::Vector4(Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<int>>::Vector4(Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<int>>::Vector4(Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<int>>::Vector4(Vector4<Complex<double>> &) noexcept;

template Vector4<Complex<long64>>::Vector4(Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<long64>>::Vector4(Vector4<Complex<double>> &) noexcept;

template Vector4<Complex<BFloat16>>::Vector4(Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<BFloat16>>::Vector4(Vector4<Complex<double>> &) noexcept;

template Vector4<Complex<HalfFp16>>::Vector4(Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(Vector4<Complex<float>> &) noexcept;
template Vector4<Complex<HalfFp16>>::Vector4(Vector4<Complex<double>> &) noexcept;

template Vector4<Complex<float>>::Vector4(Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<float>>::Vector4(Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<float>>::Vector4(Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<float>>::Vector4(Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<float>>::Vector4(Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<float>>::Vector4(Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<float>>::Vector4(Vector4<Complex<double>> &) noexcept;

template Vector4<Complex<double>>::Vector4(Vector4<Complex<sbyte>> &) noexcept;
template Vector4<Complex<double>>::Vector4(Vector4<Complex<short>> &) noexcept;
template Vector4<Complex<double>>::Vector4(Vector4<Complex<int>> &) noexcept;
template Vector4<Complex<double>>::Vector4(Vector4<Complex<long64>> &) noexcept;
template Vector4<Complex<double>>::Vector4(Vector4<Complex<BFloat16>> &) noexcept;
template Vector4<Complex<double>>::Vector4(Vector4<Complex<HalfFp16>> &) noexcept;
template Vector4<Complex<double>>::Vector4(Vector4<Complex<float>> &) noexcept;

template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<sbyte>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<byte>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<short>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<ushort>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<int>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<uint>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<long64>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<ulong64>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<float>() noexcept;
template Vector4<sbyte> &Vector4<sbyte>::ClampToTargetType<double>() noexcept;

template Vector4<byte> &Vector4<byte>::ClampToTargetType<sbyte>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<byte>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<short>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<ushort>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<int>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<uint>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<long64>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<ulong64>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<float>() noexcept;
template Vector4<byte> &Vector4<byte>::ClampToTargetType<double>() noexcept;

template Vector4<short> &Vector4<short>::ClampToTargetType<byte>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<sbyte>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<short>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<ushort>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<int>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<uint>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<long64>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<ulong64>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<float>() noexcept;
template Vector4<short> &Vector4<short>::ClampToTargetType<double>() noexcept;

template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<byte>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<sbyte>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<short>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<ushort>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<int>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<uint>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<long64>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<ulong64>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<float>() noexcept;
template Vector4<ushort> &Vector4<ushort>::ClampToTargetType<double>() noexcept;

template Vector4<int> &Vector4<int>::ClampToTargetType<byte>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<sbyte>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<short>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<ushort>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<int>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<uint>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<long64>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<ulong64>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<float>() noexcept;
template Vector4<int> &Vector4<int>::ClampToTargetType<double>() noexcept;

template Vector4<uint> &Vector4<uint>::ClampToTargetType<byte>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<sbyte>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<short>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<ushort>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<int>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<uint>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<long64>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<ulong64>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<float>() noexcept;
template Vector4<uint> &Vector4<uint>::ClampToTargetType<double>() noexcept;

template Vector4<long64> &Vector4<long64>::ClampToTargetType<byte>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<sbyte>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<short>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<ushort>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<int>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<uint>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<long64>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<ulong64>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<float>() noexcept;
template Vector4<long64> &Vector4<long64>::ClampToTargetType<double>() noexcept;

template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<byte>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<sbyte>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<short>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<ushort>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<int>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<uint>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<long64>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<ulong64>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<float>() noexcept;
template Vector4<ulong64> &Vector4<ulong64>::ClampToTargetType<double>() noexcept;

template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<byte>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<sbyte>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<short>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<ushort>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<int>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<uint>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<long64>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<ulong64>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<float>() noexcept;
template Vector4<BFloat16> &Vector4<BFloat16>::ClampToTargetType<double>() noexcept;

template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<byte>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<sbyte>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<short>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<ushort>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<int>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<uint>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<long64>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<ulong64>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<float>() noexcept;
template Vector4<HalfFp16> &Vector4<HalfFp16>::ClampToTargetType<double>() noexcept;

template Vector4<float> &Vector4<float>::ClampToTargetType<byte>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<sbyte>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<short>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<ushort>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<int>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<uint>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<long64>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<ulong64>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<float>() noexcept;
template Vector4<float> &Vector4<float>::ClampToTargetType<double>() noexcept;

template Vector4<double> &Vector4<double>::ClampToTargetType<byte>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<sbyte>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<short>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<ushort>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<int>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<uint>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<long64>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<ulong64>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<float>() noexcept;
template Vector4<double> &Vector4<double>::ClampToTargetType<double>() noexcept;

template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<sbyte>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<byte>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<short>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<ushort>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<int>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<uint>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<long64>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<ulong64>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<float>() noexcept;
template Vector4<Complex<sbyte>> &Vector4<Complex<sbyte>>::ClampToTargetType<double>() noexcept;

template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<sbyte>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<byte>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<short>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<ushort>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<int>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<uint>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<long64>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<ulong64>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<float>() noexcept;
template Vector4<Complex<short>> &Vector4<Complex<short>>::ClampToTargetType<double>() noexcept;

template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<sbyte>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<byte>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<short>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<ushort>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<int>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<uint>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<long64>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<ulong64>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<float>() noexcept;
template Vector4<Complex<int>> &Vector4<Complex<int>>::ClampToTargetType<double>() noexcept;

template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<sbyte>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<byte>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<short>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<ushort>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<int>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<uint>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<long64>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<ulong64>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<float>() noexcept;
template Vector4<Complex<long64>> &Vector4<Complex<long64>>::ClampToTargetType<double>() noexcept;

template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<sbyte>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<byte>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<short>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<ushort>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<int>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<uint>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<long64>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<ulong64>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<float>() noexcept;
template Vector4<Complex<BFloat16>> &Vector4<Complex<BFloat16>>::ClampToTargetType<double>() noexcept;

template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<sbyte>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<byte>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<short>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<ushort>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<int>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<uint>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<long64>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<ulong64>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<float>() noexcept;
template Vector4<Complex<HalfFp16>> &Vector4<Complex<HalfFp16>>::ClampToTargetType<double>() noexcept;

template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<sbyte>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<byte>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<short>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<ushort>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<int>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<uint>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<long64>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<ulong64>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<float>() noexcept;
template Vector4<Complex<float>> &Vector4<Complex<float>>::ClampToTargetType<double>() noexcept;

template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<sbyte>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<byte>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<short>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<ushort>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<int>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<uint>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<long64>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<ulong64>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<float>() noexcept;
template Vector4<Complex<double>> &Vector4<Complex<double>>::ClampToTargetType<double>() noexcept;

template std::ostream &operator<<(std::ostream &aOs, const Vector4<sbyte> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<sbyte> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<sbyte> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<sbyte> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<byte> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<byte> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<byte> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<byte> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<short> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<short> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<short> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<short> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<ushort> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<ushort> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<ushort> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<ushort> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<int> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<int> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<int> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<int> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<uint> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<uint> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<uint> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<uint> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<long64> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<long64> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<long64> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<long64> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<ulong64> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<ulong64> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<ulong64> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<ulong64> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<BFloat16> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<BFloat16> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<BFloat16> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<BFloat16> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<HalfFp16> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<HalfFp16> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<HalfFp16> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<HalfFp16> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<float> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<float> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<float> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<float> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<double> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<double> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<double> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<double> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<Complex<sbyte>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<Complex<sbyte>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<Complex<sbyte>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<Complex<sbyte>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<Complex<short>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<Complex<short>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<Complex<short>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<Complex<short>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<Complex<int>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<Complex<int>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<Complex<int>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<Complex<int>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<Complex<long64>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<Complex<long64>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<Complex<long64>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<Complex<long64>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<Complex<BFloat16>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<Complex<BFloat16>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<Complex<BFloat16>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<Complex<BFloat16>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<Complex<HalfFp16>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<Complex<HalfFp16>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<Complex<HalfFp16>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<Complex<HalfFp16>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<Complex<float>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<Complex<float>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<Complex<float>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<Complex<float>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4<Complex<double>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4<Complex<double>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4<Complex<double>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4<Complex<double>> &aVec);
} // namespace mpp