#include "vector3.h"      //NOLINT(misc-include-cleaner)
#include "bfloat16.h"     //NOLINT(misc-include-cleaner)
#include "complex.h"      //NOLINT
#include "defines.h"      //NOLINT(misc-include-cleaner)
#include "half_fp16.h"    //NOLINT(misc-include-cleaner)
#include "vector3_impl.h" //NOLINT(misc-include-cleaner)
#include <iostream>

namespace mpp
{

std::ostream &operator<<(std::ostream &aOs, const Axis3D &aAxis)
{
    switch (aAxis)
    {
        case Axis3D::X:
            aOs << 'X';
            return aOs;
        case Axis3D::Y:
            aOs << 'Y';
            return aOs;
        case Axis3D::Z:
            aOs << 'Z';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X, Y or Z (0, 1 or 2).";
    return aOs;
}

std::wostream &operator<<(std::wostream &aOs, const Axis3D &aAxis)
{
    switch (aAxis)
    {
        case Axis3D::X:
            aOs << 'X';
            return aOs;
        case Axis3D::Y:
            aOs << 'Y';
            return aOs;
        case Axis3D::Z:
            aOs << 'Z';
            return aOs;
    }
    aOs << "Out of range: " << int(aAxis) << ". Must be X, Y or Z (0, 1 or 2).";
    return aOs;
}

template struct Vector3<sbyte>;
template struct Vector3<byte>;
template struct Vector3<short>;
template struct Vector3<ushort>;
template struct Vector3<int>;
template struct Vector3<uint>;
template struct Vector3<long64>;
template struct Vector3<ulong64>;

template struct Vector3<BFloat16>;
template struct Vector3<HalfFp16>;
template struct Vector3<float>;
template struct Vector3<double>;

template struct Vector3<Complex<sbyte>>;
template struct Vector3<Complex<short>>;
template struct Vector3<Complex<int>>;
template struct Vector3<Complex<BFloat16>>;
template struct Vector3<Complex<HalfFp16>>;
template struct Vector3<Complex<float>>;
template struct Vector3<Complex<double>>;

template Vector3<sbyte>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<short> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<int> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<float> &) noexcept;
template Vector3<sbyte>::Vector3(const Vector3<double> &) noexcept;

template Vector3<byte>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<short> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<int> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<float> &) noexcept;
template Vector3<byte>::Vector3(const Vector3<double> &) noexcept;

template Vector3<short>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<short>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<short>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<short>::Vector3(const Vector3<int> &) noexcept;
template Vector3<short>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<short>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<short>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<short>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<short>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<short>::Vector3(const Vector3<float> &) noexcept;
template Vector3<short>::Vector3(const Vector3<double> &) noexcept;

template Vector3<ushort>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<short> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<int> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<float> &) noexcept;
template Vector3<ushort>::Vector3(const Vector3<double> &) noexcept;

template Vector3<int>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<int>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<int>::Vector3(const Vector3<short> &) noexcept;
template Vector3<int>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<int>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<int>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<int>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<int>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<int>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<int>::Vector3(const Vector3<float> &) noexcept;
template Vector3<int>::Vector3(const Vector3<double> &) noexcept;

template Vector3<uint>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<short> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<int> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<float> &) noexcept;
template Vector3<uint>::Vector3(const Vector3<double> &) noexcept;

template Vector3<long64>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<short> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<int> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<float> &) noexcept;
template Vector3<long64>::Vector3(const Vector3<double> &) noexcept;

template Vector3<ulong64>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<short> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<int> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<float> &) noexcept;
template Vector3<ulong64>::Vector3(const Vector3<double> &) noexcept;

template Vector3<BFloat16>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<short> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<int> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<float> &) noexcept;
template Vector3<BFloat16>::Vector3(const Vector3<double> &) noexcept;

template Vector3<HalfFp16>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<short> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<int> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<float> &) noexcept;
template Vector3<HalfFp16>::Vector3(const Vector3<double> &) noexcept;

template Vector3<float>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<float>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<float>::Vector3(const Vector3<short> &) noexcept;
template Vector3<float>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<float>::Vector3(const Vector3<int> &) noexcept;
template Vector3<float>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<float>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<float>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<float>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<float>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<float>::Vector3(const Vector3<double> &) noexcept;

template Vector3<double>::Vector3(const Vector3<sbyte> &) noexcept;
template Vector3<double>::Vector3(const Vector3<byte> &) noexcept;
template Vector3<double>::Vector3(const Vector3<short> &) noexcept;
template Vector3<double>::Vector3(const Vector3<ushort> &) noexcept;
template Vector3<double>::Vector3(const Vector3<int> &) noexcept;
template Vector3<double>::Vector3(const Vector3<uint> &) noexcept;
template Vector3<double>::Vector3(const Vector3<long64> &) noexcept;
template Vector3<double>::Vector3(const Vector3<ulong64> &) noexcept;
template Vector3<double>::Vector3(const Vector3<BFloat16> &) noexcept;
template Vector3<double>::Vector3(const Vector3<HalfFp16> &) noexcept;
template Vector3<double>::Vector3(const Vector3<float> &) noexcept;

template Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<double>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(const Vector3<sbyte> &) noexcept;

template Vector3<Complex<short>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<short>>::Vector3(const Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<short>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<short>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<short>>::Vector3(const Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<short>>::Vector3(const Vector3<Complex<double>> &) noexcept;
template Vector3<Complex<short>>::Vector3(const Vector3<short> &) noexcept;

template Vector3<Complex<int>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<int>>::Vector3(const Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<int>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<int>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<int>>::Vector3(const Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<int>>::Vector3(const Vector3<Complex<double>> &) noexcept;
template Vector3<Complex<int>>::Vector3(const Vector3<int> &) noexcept;

template Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<double>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(const Vector3<BFloat16> &) noexcept;

template Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<double>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(const Vector3<HalfFp16> &) noexcept;

template Vector3<Complex<float>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<float>>::Vector3(const Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<float>>::Vector3(const Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<float>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<float>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<float>>::Vector3(const Vector3<Complex<double>> &) noexcept;
template Vector3<Complex<float>>::Vector3(const Vector3<float> &) noexcept;

template Vector3<Complex<double>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<double>>::Vector3(const Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<double>>::Vector3(const Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<double>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<double>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<double>>::Vector3(const Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<double>>::Vector3(const Vector3<double> &) noexcept;

template Vector3<sbyte>::Vector3(Vector3<byte> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<short> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<int> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<uint> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<long64> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<float> &) noexcept;
template Vector3<sbyte>::Vector3(Vector3<double> &) noexcept;

template Vector3<byte>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<byte>::Vector3(Vector3<short> &) noexcept;
template Vector3<byte>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<byte>::Vector3(Vector3<int> &) noexcept;
template Vector3<byte>::Vector3(Vector3<uint> &) noexcept;
template Vector3<byte>::Vector3(Vector3<long64> &) noexcept;
template Vector3<byte>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<byte>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<byte>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<byte>::Vector3(Vector3<float> &) noexcept;
template Vector3<byte>::Vector3(Vector3<double> &) noexcept;

template Vector3<short>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<short>::Vector3(Vector3<byte> &) noexcept;
template Vector3<short>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<short>::Vector3(Vector3<int> &) noexcept;
template Vector3<short>::Vector3(Vector3<uint> &) noexcept;
template Vector3<short>::Vector3(Vector3<long64> &) noexcept;
template Vector3<short>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<short>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<short>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<short>::Vector3(Vector3<float> &) noexcept;
template Vector3<short>::Vector3(Vector3<double> &) noexcept;

template Vector3<ushort>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<byte> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<short> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<int> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<uint> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<long64> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<float> &) noexcept;
template Vector3<ushort>::Vector3(Vector3<double> &) noexcept;

template Vector3<int>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<int>::Vector3(Vector3<byte> &) noexcept;
template Vector3<int>::Vector3(Vector3<short> &) noexcept;
template Vector3<int>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<int>::Vector3(Vector3<uint> &) noexcept;
template Vector3<int>::Vector3(Vector3<long64> &) noexcept;
template Vector3<int>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<int>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<int>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<int>::Vector3(Vector3<float> &) noexcept;
template Vector3<int>::Vector3(Vector3<double> &) noexcept;

template Vector3<uint>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<uint>::Vector3(Vector3<byte> &) noexcept;
template Vector3<uint>::Vector3(Vector3<short> &) noexcept;
template Vector3<uint>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<uint>::Vector3(Vector3<int> &) noexcept;
template Vector3<uint>::Vector3(Vector3<long64> &) noexcept;
template Vector3<uint>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<uint>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<uint>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<uint>::Vector3(Vector3<float> &) noexcept;
template Vector3<uint>::Vector3(Vector3<double> &) noexcept;

template Vector3<long64>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<long64>::Vector3(Vector3<byte> &) noexcept;
template Vector3<long64>::Vector3(Vector3<short> &) noexcept;
template Vector3<long64>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<long64>::Vector3(Vector3<int> &) noexcept;
template Vector3<long64>::Vector3(Vector3<uint> &) noexcept;
template Vector3<long64>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<long64>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<long64>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<long64>::Vector3(Vector3<float> &) noexcept;
template Vector3<long64>::Vector3(Vector3<double> &) noexcept;

template Vector3<ulong64>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<byte> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<short> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<int> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<uint> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<long64> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<float> &) noexcept;
template Vector3<ulong64>::Vector3(Vector3<double> &) noexcept;

template Vector3<BFloat16>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<byte> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<short> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<int> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<uint> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<long64> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<float> &) noexcept;
template Vector3<BFloat16>::Vector3(Vector3<double> &) noexcept;

template Vector3<HalfFp16>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<byte> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<short> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<int> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<uint> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<long64> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<float> &) noexcept;
template Vector3<HalfFp16>::Vector3(Vector3<double> &) noexcept;

template Vector3<float>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<float>::Vector3(Vector3<byte> &) noexcept;
template Vector3<float>::Vector3(Vector3<short> &) noexcept;
template Vector3<float>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<float>::Vector3(Vector3<int> &) noexcept;
template Vector3<float>::Vector3(Vector3<uint> &) noexcept;
template Vector3<float>::Vector3(Vector3<long64> &) noexcept;
template Vector3<float>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<float>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<float>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<float>::Vector3(Vector3<double> &) noexcept;

template Vector3<double>::Vector3(Vector3<sbyte> &) noexcept;
template Vector3<double>::Vector3(Vector3<byte> &) noexcept;
template Vector3<double>::Vector3(Vector3<short> &) noexcept;
template Vector3<double>::Vector3(Vector3<ushort> &) noexcept;
template Vector3<double>::Vector3(Vector3<int> &) noexcept;
template Vector3<double>::Vector3(Vector3<uint> &) noexcept;
template Vector3<double>::Vector3(Vector3<long64> &) noexcept;
template Vector3<double>::Vector3(Vector3<ulong64> &) noexcept;
template Vector3<double>::Vector3(Vector3<BFloat16> &) noexcept;
template Vector3<double>::Vector3(Vector3<HalfFp16> &) noexcept;
template Vector3<double>::Vector3(Vector3<float> &) noexcept;

template Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<double>> &) noexcept;

template Vector3<Complex<short>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<short>>::Vector3(Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<short>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<short>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<short>>::Vector3(Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<short>>::Vector3(Vector3<Complex<double>> &) noexcept;

template Vector3<Complex<int>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<int>>::Vector3(Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<int>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<int>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<int>>::Vector3(Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<int>>::Vector3(Vector3<Complex<double>> &) noexcept;

template Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<double>> &) noexcept;

template Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<float>> &) noexcept;
template Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<double>> &) noexcept;

template Vector3<Complex<float>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<float>>::Vector3(Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<float>>::Vector3(Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<float>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<float>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<float>>::Vector3(Vector3<Complex<double>> &) noexcept;

template Vector3<Complex<double>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
template Vector3<Complex<double>>::Vector3(Vector3<Complex<short>> &) noexcept;
template Vector3<Complex<double>>::Vector3(Vector3<Complex<int>> &) noexcept;
template Vector3<Complex<double>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
template Vector3<Complex<double>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
template Vector3<Complex<double>>::Vector3(Vector3<Complex<float>> &) noexcept;

template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<sbyte>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<byte>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<short>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<ushort>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<int>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<uint>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<long64>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<ulong64>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<float>() noexcept;
template Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<double>() noexcept;

template Vector3<byte> &Vector3<byte>::ClampToTargetType<sbyte>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<byte>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<short>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<ushort>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<int>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<uint>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<long64>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<ulong64>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<float>() noexcept;
template Vector3<byte> &Vector3<byte>::ClampToTargetType<double>() noexcept;

template Vector3<short> &Vector3<short>::ClampToTargetType<byte>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<sbyte>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<short>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<ushort>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<int>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<uint>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<long64>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<ulong64>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<float>() noexcept;
template Vector3<short> &Vector3<short>::ClampToTargetType<double>() noexcept;

template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<byte>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<sbyte>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<short>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<ushort>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<int>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<uint>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<long64>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<ulong64>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<float>() noexcept;
template Vector3<ushort> &Vector3<ushort>::ClampToTargetType<double>() noexcept;

template Vector3<int> &Vector3<int>::ClampToTargetType<byte>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<sbyte>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<short>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<ushort>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<int>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<uint>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<long64>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<ulong64>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<float>() noexcept;
template Vector3<int> &Vector3<int>::ClampToTargetType<double>() noexcept;

template Vector3<uint> &Vector3<uint>::ClampToTargetType<byte>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<sbyte>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<short>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<ushort>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<int>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<uint>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<long64>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<ulong64>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<float>() noexcept;
template Vector3<uint> &Vector3<uint>::ClampToTargetType<double>() noexcept;

template Vector3<long64> &Vector3<long64>::ClampToTargetType<byte>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<sbyte>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<short>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<ushort>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<int>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<uint>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<long64>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<ulong64>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<float>() noexcept;
template Vector3<long64> &Vector3<long64>::ClampToTargetType<double>() noexcept;

template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<byte>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<sbyte>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<short>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<ushort>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<int>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<uint>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<long64>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<ulong64>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<float>() noexcept;
template Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<double>() noexcept;

template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<byte>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<sbyte>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<short>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<ushort>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<int>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<uint>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<long64>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<ulong64>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<float>() noexcept;
template Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<double>() noexcept;

template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<byte>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<sbyte>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<short>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<ushort>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<int>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<uint>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<long64>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<ulong64>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<float>() noexcept;
template Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<double>() noexcept;

template Vector3<float> &Vector3<float>::ClampToTargetType<byte>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<sbyte>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<short>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<ushort>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<int>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<uint>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<long64>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<ulong64>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<float>() noexcept;
template Vector3<float> &Vector3<float>::ClampToTargetType<double>() noexcept;

template Vector3<double> &Vector3<double>::ClampToTargetType<byte>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<sbyte>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<short>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<ushort>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<int>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<uint>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<long64>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<ulong64>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<float>() noexcept;
template Vector3<double> &Vector3<double>::ClampToTargetType<double>() noexcept;

template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<sbyte>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<byte>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<short>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<ushort>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<int>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<uint>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<long64>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<ulong64>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<float>() noexcept;
template Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<double>() noexcept;

template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<sbyte>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<byte>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<short>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<ushort>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<int>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<uint>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<long64>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<ulong64>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<float>() noexcept;
template Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<double>() noexcept;

template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<sbyte>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<byte>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<short>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<ushort>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<int>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<uint>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<long64>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<ulong64>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<float>() noexcept;
template Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<double>() noexcept;

template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<sbyte>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<byte>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<short>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<ushort>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<int>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<uint>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<long64>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<ulong64>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<float>() noexcept;
template Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<double>() noexcept;

template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<sbyte>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<byte>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<short>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<ushort>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<int>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<uint>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<long64>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<ulong64>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<float>() noexcept;
template Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<double>() noexcept;

template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<sbyte>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<byte>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<short>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<ushort>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<int>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<uint>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<long64>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<ulong64>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<float>() noexcept;
template Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<double>() noexcept;

template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<sbyte>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<byte>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<short>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<ushort>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<int>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<uint>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<long64>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<ulong64>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<float>() noexcept;
template Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<double>() noexcept;

template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<sbyte>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<byte>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<short>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<ushort>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<int>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<uint>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<long64>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<ulong64>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<BFloat16>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<float>() noexcept;
template Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<double>() noexcept;

template std::ostream &operator<<(std::ostream &aOs, const Vector3<sbyte> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<sbyte> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<sbyte> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<sbyte> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<byte> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<byte> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<byte> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<byte> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<short> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<short> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<short> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<short> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<ushort> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<ushort> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<ushort> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<ushort> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<int> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<int> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<int> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<int> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<uint> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<uint> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<uint> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<uint> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<long64> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<long64> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<long64> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<long64> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<ulong64> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<ulong64> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<ulong64> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<ulong64> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<BFloat16> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<BFloat16> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<BFloat16> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<BFloat16> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<HalfFp16> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<HalfFp16> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<HalfFp16> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<HalfFp16> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<float> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<float> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<float> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<float> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<double> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<double> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<double> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<double> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<sbyte>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<sbyte>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<Complex<sbyte>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<sbyte>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<short>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<short>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<Complex<short>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<short>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<int>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<int>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<Complex<int>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<int>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<BFloat16>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<BFloat16>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<Complex<BFloat16>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<BFloat16>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<HalfFp16>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<HalfFp16>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<Complex<HalfFp16>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<HalfFp16>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<float>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<float>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<Complex<float>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<float>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<double>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<double>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector3<Complex<double>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<double>> &aVec);
} // namespace mpp