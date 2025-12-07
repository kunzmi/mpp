#include "bfloat16.h" //NOLINT(misc-include-cleaner)
#include "complex.h"  //NOLINT
#include "defines.h"  //NOLINT(misc-include-cleaner)
#include "dllexport_common.h"
#include "half_fp16.h"    //NOLINT(misc-include-cleaner)
#include "vector2.h"  //NOLINT(misc-include-cleaner)
#include "vector2_impl.h" //NOLINT(misc-include-cleaner)
#include <iostream>

namespace mpp
{

std::ostream &operator<<(std::ostream &aOs, const Axis2D &aAxis)
{
    switch (aAxis)
    {
        case Axis2D::X:
            aOs << 'X';
            return aOs;
        case Axis2D::Y:
            aOs << 'Y';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X or Y (0 or 1).";
    return aOs;
}

std::wostream &operator<<(std::wostream &aOs, const Axis2D &aAxis)
{
    switch (aAxis)
    {
        case Axis2D::X:
            aOs << 'X';
            return aOs;
        case Axis2D::Y:
            aOs << 'Y';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X or Y (0 or 1).";
    return aOs;
}

template struct Vector2<sbyte>;
template struct Vector2<byte>;
template struct Vector2<short>;
template struct Vector2<ushort>;
template struct Vector2<int>;
template struct Vector2<uint>;
template struct Vector2<long64>;
template struct Vector2<ulong64>;

template struct Vector2<BFloat16>;
template struct Vector2<HalfFp16>;
template struct Vector2<float>;
template struct Vector2<double>;

template struct Vector2<Complex<sbyte>>;
template struct Vector2<Complex<short>>;
template struct Vector2<Complex<int>>;
template struct Vector2<Complex<long64>>;
template struct Vector2<Complex<BFloat16>>;
template struct Vector2<Complex<HalfFp16>>;
template struct Vector2<Complex<float>>;
template struct Vector2<Complex<double>>;

template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<double> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(const Vector2<float> &, RoundingMode);

template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<double> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(const Vector2<float> &, RoundingMode);

template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(const Vector2<float> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<Complex<double>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(const Vector2<sbyte> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<Complex<double>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(const Vector2<short> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<Complex<double>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(const Vector2<int> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<Complex<double>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(const Vector2<long64> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<double>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(const Vector2<Complex<float>> &, RoundingMode);

template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<double>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(const Vector2<Complex<float>> &, RoundingMode);

template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<Complex<double>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(const Vector2<float> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(const Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<sbyte>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<byte>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<short>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<ushort>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<int>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<uint>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<long64>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<ulong64>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<float> &) noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<float>::Vector2(Vector2<double> &) noexcept;

template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<sbyte> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<byte> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<short> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<ushort> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<int> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<uint> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<long64> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<ulong64> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<BFloat16> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<HalfFp16> &) noexcept;
template MPPEXPORT_COMMON Vector2<double>::Vector2(Vector2<float> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>>::Vector2(Vector2<Complex<double>> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>>::Vector2(Vector2<Complex<double>> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>>::Vector2(Vector2<Complex<double>> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>>::Vector2(Vector2<Complex<double>> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>>::Vector2(Vector2<Complex<double>> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<float>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>>::Vector2(Vector2<Complex<double>> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>>::Vector2(Vector2<Complex<double>> &) noexcept;

template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<sbyte>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<short>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<int>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<long64>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<BFloat16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<HalfFp16>> &) noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>>::Vector2(Vector2<Complex<float>> &) noexcept;

template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<sbyte> &Vector2<sbyte>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<byte> &Vector2<byte>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<short> &Vector2<short>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<ushort> &Vector2<ushort>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<int> &Vector2<int>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<uint> &Vector2<uint>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<long64> &Vector2<long64>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<ulong64> &Vector2<ulong64>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<BFloat16> &Vector2<BFloat16>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<HalfFp16> &Vector2<HalfFp16>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<float> &Vector2<float>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<double> &Vector2<double>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<sbyte>> &Vector2<Complex<sbyte>>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<short>> &Vector2<Complex<short>>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<int>> &Vector2<Complex<int>>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<long64>> &Vector2<Complex<long64>>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<
    HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<BFloat16>> &Vector2<Complex<BFloat16>>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<
    HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<HalfFp16>> &Vector2<Complex<HalfFp16>>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<float>> &Vector2<Complex<float>>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<sbyte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<byte>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<short>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<ushort>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<int>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<uint>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<long64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<ulong64>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<BFloat16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<HalfFp16>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<float>() noexcept;
template MPPEXPORT_COMMON Vector2<Complex<double>> &Vector2<Complex<double>>::ClampToTargetType<double>() noexcept;

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<sbyte> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<sbyte> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<sbyte> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<sbyte> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<byte> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<byte> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<byte> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<byte> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<short> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<short> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<short> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<short> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<ushort> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<ushort> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<ushort> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<ushort> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<int> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<int> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<int> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<int> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<uint> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<uint> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<uint> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<uint> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<long64> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<long64> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<long64> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<long64> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<ulong64> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<ulong64> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<ulong64> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<ulong64> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<BFloat16> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<BFloat16> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<BFloat16> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<BFloat16> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<HalfFp16> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<HalfFp16> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<HalfFp16> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<HalfFp16> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<float> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<float> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<float> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<float> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<double> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<double> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<double> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<double> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<sbyte>> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<sbyte>> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<sbyte>> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<sbyte>> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<short>> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<short>> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<short>> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<short>> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<int>> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<int>> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<int>> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<int>> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<long64>> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<long64>> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<long64>> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<long64>> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<BFloat16>> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<BFloat16>> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<BFloat16>> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<BFloat16>> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<HalfFp16>> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<HalfFp16>> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<HalfFp16>> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<HalfFp16>> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<float>> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<float>> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<float>> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<float>> &aVec);

template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector2<Complex<double>> &aVec);
template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector2<Complex<double>> &aVec);
template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector2<Complex<double>> &aVec);
template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector2<Complex<double>> &aVec);
} // namespace mpp