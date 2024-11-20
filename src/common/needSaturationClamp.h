#pragma once
#include "defines.h"
#include "limits.h"
#include <type_traits>

// define which type conversion from->to need value clamping. E.g. an integer of value 1024 cannot be stored in byte and
// must thus be clamped to 255
namespace opp
{
// by default, no clamping needed
template <typename TFrom, typename TTo> struct need_saturation_clamp : std::false_type
{
};

template <> struct need_saturation_clamp<byte, sbyte> : std::true_type
{
};

template <> struct need_saturation_clamp<sbyte, byte> : std::true_type
{
};
template <> struct need_saturation_clamp<sbyte, ushort> : std::true_type
{
};
template <> struct need_saturation_clamp<sbyte, uint> : std::true_type
{
};
template <> struct need_saturation_clamp<sbyte, ulong64> : std::true_type
{
};

template <> struct need_saturation_clamp<short, byte> : std::true_type
{
};
template <> struct need_saturation_clamp<short, sbyte> : std::true_type
{
};
template <> struct need_saturation_clamp<short, ushort> : std::true_type
{
};
template <> struct need_saturation_clamp<short, uint> : std::true_type
{
};
template <> struct need_saturation_clamp<short, ulong64> : std::true_type
{
};

template <> struct need_saturation_clamp<ushort, byte> : std::true_type
{
};
template <> struct need_saturation_clamp<ushort, sbyte> : std::true_type
{
};
template <> struct need_saturation_clamp<ushort, short> : std::true_type
{
};

template <> struct need_saturation_clamp<int, byte> : std::true_type
{
};
template <> struct need_saturation_clamp<int, sbyte> : std::true_type
{
};
template <> struct need_saturation_clamp<int, short> : std::true_type
{
};
template <> struct need_saturation_clamp<int, ushort> : std::true_type
{
};
template <> struct need_saturation_clamp<int, uint> : std::true_type
{
};
template <> struct need_saturation_clamp<int, ulong64> : std::true_type
{
};

template <> struct need_saturation_clamp<uint, byte> : std::true_type
{
};
template <> struct need_saturation_clamp<uint, sbyte> : std::true_type
{
};
template <> struct need_saturation_clamp<uint, short> : std::true_type
{
};
template <> struct need_saturation_clamp<uint, ushort> : std::true_type
{
};
template <> struct need_saturation_clamp<uint, int> : std::true_type
{
};

template <> struct need_saturation_clamp<long64, byte> : std::true_type
{
};
template <> struct need_saturation_clamp<long64, sbyte> : std::true_type
{
};
template <> struct need_saturation_clamp<long64, short> : std::true_type
{
};
template <> struct need_saturation_clamp<long64, ushort> : std::true_type
{
};
template <> struct need_saturation_clamp<long64, int> : std::true_type
{
};
template <> struct need_saturation_clamp<long64, uint> : std::true_type
{
};
template <> struct need_saturation_clamp<long64, ulong64> : std::true_type
{
};

template <> struct need_saturation_clamp<ulong64, byte> : std::true_type
{
};
template <> struct need_saturation_clamp<ulong64, sbyte> : std::true_type
{
};
template <> struct need_saturation_clamp<ulong64, short> : std::true_type
{
};
template <> struct need_saturation_clamp<ulong64, ushort> : std::true_type
{
};
template <> struct need_saturation_clamp<ulong64, int> : std::true_type
{
};
template <> struct need_saturation_clamp<ulong64, uint> : std::true_type
{
};
template <> struct need_saturation_clamp<ulong64, long64> : std::true_type
{
};

template <> struct need_saturation_clamp<float, byte> : std::true_type
{
};
template <> struct need_saturation_clamp<float, sbyte> : std::true_type
{
};
template <> struct need_saturation_clamp<float, short> : std::true_type
{
};
template <> struct need_saturation_clamp<float, ushort> : std::true_type
{
};
template <> struct need_saturation_clamp<float, int> : std::true_type
{
};
template <> struct need_saturation_clamp<float, uint> : std::true_type
{
};
template <> struct need_saturation_clamp<float, long64> : std::true_type
{
};
template <> struct need_saturation_clamp<float, ulong64> : std::true_type
{
};

template <> struct need_saturation_clamp<double, byte> : std::true_type
{
};
template <> struct need_saturation_clamp<double, sbyte> : std::true_type
{
};
template <> struct need_saturation_clamp<double, short> : std::true_type
{
};
template <> struct need_saturation_clamp<double, ushort> : std::true_type
{
};
template <> struct need_saturation_clamp<double, int> : std::true_type
{
};
template <> struct need_saturation_clamp<double, uint> : std::true_type
{
};
template <> struct need_saturation_clamp<double, long64> : std::true_type
{
};
template <> struct need_saturation_clamp<double, ulong64> : std::true_type
{
};

} // namespace opp