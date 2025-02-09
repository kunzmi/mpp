#include "vector4A.h"      //NOLINT(misc-include-cleaner)
#include "bfloat16.h"      //NOLINT(misc-include-cleaner)
#include "complex.h"       //NOLINT
#include "defines.h"       //NOLINT(misc-include-cleaner)
#include "half_fp16.h"     //NOLINT(misc-include-cleaner)
#include "vector4A_impl.h" //NOLINT(misc-include-cleaner)

namespace opp
{
template struct Vector4A<sbyte>;
template struct Vector4A<byte>;
template struct Vector4A<short>;
template struct Vector4A<ushort>;
template struct Vector4A<int>;
template struct Vector4A<uint>;
template struct Vector4A<long64>;
template struct Vector4A<ulong64>;

template struct Vector4A<BFloat16>;
template struct Vector4A<HalfFp16>;
template struct Vector4A<float>;
template struct Vector4A<double>;

template struct Vector4A<Complex<sbyte>>;
template struct Vector4A<Complex<short>>;
template struct Vector4A<Complex<int>>;
template struct Vector4A<Complex<BFloat16>>;
template struct Vector4A<Complex<HalfFp16>>;
template struct Vector4A<Complex<float>>;
template struct Vector4A<Complex<double>>;

template Vector4A<sbyte>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<sbyte>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<byte>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<byte>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<short>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<short>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<ushort>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<ushort>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<int>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<int>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<uint>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<uint>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<long64>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<long64>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<ulong64>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<ulong64>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<BFloat16>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<BFloat16>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<HalfFp16>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<float> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<float>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<float>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<double>::Vector4A(const Vector4A<sbyte> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<byte> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<short> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<ushort> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<int> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<uint> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<long64> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<ulong64> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<BFloat16> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<HalfFp16> &) noexcept;
template Vector4A<double>::Vector4A(const Vector4A<float> &) noexcept;

template Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(const Vector4A<sbyte> &) noexcept;

template Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(const Vector4A<short> &) noexcept;

template Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(const Vector4A<int> &) noexcept;

template Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(const Vector4A<BFloat16> &) noexcept;

template Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(const Vector4A<HalfFp16> &) noexcept;

template Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(const Vector4A<Complex<double>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(const Vector4A<float> &) noexcept;

template Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(const Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(const Vector4A<double> &) noexcept;

template Vector4A<sbyte>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<sbyte>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<byte>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<byte>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<short>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<short>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<ushort>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<ushort>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<int>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<int>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<uint>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<uint>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<long64>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<long64>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<ulong64>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<ulong64>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<BFloat16>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<BFloat16>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<HalfFp16>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<float> &) noexcept;
template Vector4A<HalfFp16>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<float>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<float>::Vector4A(Vector4A<double> &) noexcept;

template Vector4A<double>::Vector4A(Vector4A<sbyte> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<byte> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<short> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<ushort> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<int> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<uint> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<long64> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<ulong64> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<BFloat16> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<HalfFp16> &) noexcept;
template Vector4A<double>::Vector4A(Vector4A<float> &) noexcept;

template Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<sbyte>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

template Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<short>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

template Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<int>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

template Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<BFloat16>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

template Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<float>> &) noexcept;
template Vector4A<Complex<HalfFp16>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

template Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<float>>::Vector4A(Vector4A<Complex<double>> &) noexcept;

template Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<sbyte>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<short>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<int>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<BFloat16>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<HalfFp16>> &) noexcept;
template Vector4A<Complex<double>>::Vector4A(Vector4A<Complex<float>> &) noexcept;

template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<byte>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<short>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<ushort>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<int>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<uint>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<long64>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<float>() noexcept;
template Vector4A<sbyte> &Vector4A<sbyte>::ClampToTargetType<double>() noexcept;

template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<byte>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<short>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<ushort>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<int>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<uint>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<long64>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<float>() noexcept;
template Vector4A<byte> &Vector4A<byte>::ClampToTargetType<double>() noexcept;

template Vector4A<short> &Vector4A<short>::ClampToTargetType<byte>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<short>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<ushort>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<int>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<uint>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<long64>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<float>() noexcept;
template Vector4A<short> &Vector4A<short>::ClampToTargetType<double>() noexcept;

template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<byte>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<short>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<ushort>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<int>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<uint>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<long64>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<float>() noexcept;
template Vector4A<ushort> &Vector4A<ushort>::ClampToTargetType<double>() noexcept;

template Vector4A<int> &Vector4A<int>::ClampToTargetType<byte>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<short>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<ushort>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<int>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<uint>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<long64>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<float>() noexcept;
template Vector4A<int> &Vector4A<int>::ClampToTargetType<double>() noexcept;

template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<byte>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<short>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<ushort>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<int>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<uint>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<long64>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<float>() noexcept;
template Vector4A<uint> &Vector4A<uint>::ClampToTargetType<double>() noexcept;

template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<byte>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<short>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<ushort>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<int>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<uint>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<long64>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<float>() noexcept;
template Vector4A<long64> &Vector4A<long64>::ClampToTargetType<double>() noexcept;

template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<byte>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<short>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<ushort>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<int>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<uint>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<long64>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<float>() noexcept;
template Vector4A<ulong64> &Vector4A<ulong64>::ClampToTargetType<double>() noexcept;

template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<byte>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<short>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<ushort>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<int>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<uint>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<long64>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<float>() noexcept;
template Vector4A<BFloat16> &Vector4A<BFloat16>::ClampToTargetType<double>() noexcept;

template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<byte>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<short>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<ushort>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<int>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<uint>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<long64>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<float>() noexcept;
template Vector4A<HalfFp16> &Vector4A<HalfFp16>::ClampToTargetType<double>() noexcept;

template Vector4A<float> &Vector4A<float>::ClampToTargetType<byte>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<short>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<ushort>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<int>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<uint>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<long64>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<float>() noexcept;
template Vector4A<float> &Vector4A<float>::ClampToTargetType<double>() noexcept;

template Vector4A<double> &Vector4A<double>::ClampToTargetType<byte>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<short>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<ushort>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<int>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<uint>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<long64>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<float>() noexcept;
template Vector4A<double> &Vector4A<double>::ClampToTargetType<double>() noexcept;

template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<byte>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<short>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<ushort>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<int>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<uint>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<long64>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<float>() noexcept;
template Vector4A<Complex<sbyte>> &Vector4A<Complex<sbyte>>::ClampToTargetType<double>() noexcept;

template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<byte>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<short>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<ushort>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<int>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<uint>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<long64>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<float>() noexcept;
template Vector4A<Complex<short>> &Vector4A<Complex<short>>::ClampToTargetType<double>() noexcept;

template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<byte>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<short>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<ushort>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<int>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<uint>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<long64>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<float>() noexcept;
template Vector4A<Complex<int>> &Vector4A<Complex<int>>::ClampToTargetType<double>() noexcept;

template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<byte>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<short>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<ushort>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<int>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<uint>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<long64>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<float>() noexcept;
template Vector4A<Complex<long64>> &Vector4A<Complex<long64>>::ClampToTargetType<double>() noexcept;

template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<byte>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<short>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<ushort>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<int>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<uint>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<long64>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<float>() noexcept;
template Vector4A<Complex<BFloat16>> &Vector4A<Complex<BFloat16>>::ClampToTargetType<double>() noexcept;

template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<byte>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<short>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<ushort>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<int>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<uint>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<long64>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<float>() noexcept;
template Vector4A<Complex<HalfFp16>> &Vector4A<Complex<HalfFp16>>::ClampToTargetType<double>() noexcept;

template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<byte>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<short>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<ushort>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<int>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<uint>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<long64>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<float>() noexcept;
template Vector4A<Complex<float>> &Vector4A<Complex<float>>::ClampToTargetType<double>() noexcept;

template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<sbyte>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<byte>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<short>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<ushort>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<int>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<uint>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<long64>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<ulong64>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<BFloat16>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<HalfFp16>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<float>() noexcept;
template Vector4A<Complex<double>> &Vector4A<Complex<double>>::ClampToTargetType<double>() noexcept;

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<sbyte> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<sbyte> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<sbyte> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<sbyte> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<byte> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<byte> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<byte> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<byte> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<short> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<short> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<short> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<short> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<ushort> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<ushort> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<ushort> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<ushort> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<int> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<int> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<int> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<int> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<uint> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<uint> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<uint> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<uint> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<long64> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<long64> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<long64> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<long64> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<ulong64> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<ulong64> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<ulong64> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<ulong64> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<BFloat16> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<BFloat16> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<BFloat16> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<BFloat16> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<HalfFp16> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<HalfFp16> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<HalfFp16> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<HalfFp16> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<float> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<float> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<float> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<float> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<double> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<double> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<double> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<double> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<sbyte>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<sbyte>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<Complex<sbyte>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<sbyte>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<short>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<short>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<Complex<short>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<short>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<int>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<int>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<Complex<int>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<int>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<BFloat16>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<BFloat16>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<Complex<BFloat16>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<BFloat16>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<HalfFp16>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<HalfFp16>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<Complex<HalfFp16>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<HalfFp16>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<float>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<float>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<Complex<float>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<float>> &aVec);

template std::ostream &operator<<(std::ostream &aOs, const Vector4A<Complex<double>> &aVec);
template std::wostream &operator<<(std::wostream &aOs, const Vector4A<Complex<double>> &aVec);
template std::istream &operator>>(std::istream &aIs, Vector4A<Complex<double>> &aVec);
template std::wistream &operator>>(std::wistream &aIs, Vector4A<Complex<double>> &aVec);
} // namespace opp