#pragma once
#include "tupel.h"
#include <common/defines.h>
#include <concepts>
#include <cstddef>

namespace opp
{

// declare the CUDA intrinsic used later
uint __vaddus4(uint, uint);

/// <summary>
/// A spezialised Tupel for masks.
/// </summary>
/// <typeparam name="size">Number of elements, must be &lt;= 64 for Load and Store functions</typeparam>
template <std::size_t size> struct MaskTupel : public Tupel<byte, size>
{
  public:
    MaskTupel()  = default;
    ~MaskTupel() = default;

    MaskTupel(const MaskTupel &)     = default;
    MaskTupel(MaskTupel &&) noexcept = default;

    MaskTupel &operator=(const MaskTupel &)     = default;
    MaskTupel &operator=(MaskTupel &&) noexcept = default;

    MaskTupel(const Tupel<byte, size> &aOther) : Tupel<byte, size>(aOther)
    {
    }
    MaskTupel(Tupel<byte, size> &&aOther) noexcept : Tupel<byte, size>(aOther)
    {
    }

    MaskTupel &operator=(const Tupel<byte, size> &aOther)
    {
        Tupel<byte, size>::value = aOther.value;
    }
    MaskTupel &operator=(Tupel<byte, size> &&aOther) noexcept
    {
        Tupel<byte, size>::value = aOther.value;
    }

    /// <summary>
    /// Useful helper method when using masks
    /// </summary>
    bool DEVICE_CODE AreAllFalse()
        requires(size == 1)
    {
        return Tupel<byte, size>::value[0] == static_cast<byte>(0);
    } // namespace opp

    /// <summary>
    /// Useful helper method when using masks
    /// </summary>
    bool DEVICE_CODE AreAllFalse()
        requires(size == 2)
    {
        ushort &us = *reinterpret_cast<ushort *>(this);
        return us == static_cast<ushort>(0);
    }

    /// <summary>
    /// Useful helper method when using masks
    /// </summary>
    bool DEVICE_CODE AreAllFalse()
        requires(size == 4)
    {
        uint &ui = *reinterpret_cast<uint *>(this);
        return ui == static_cast<uint>(0);
    }

    /// <summary>
    /// Useful helper method when using masks
    /// </summary>
    bool DEVICE_CODE AreAllFalse()
        requires(size == 8)
    {
        ulong64 &ul = *reinterpret_cast<ulong64 *>(this);
        return ul == static_cast<ulong64>(0);
    }

    /// <summary>
    /// Useful helper method when using masks
    /// </summary>
    bool DEVICE_CODE AreAllFalse()
        requires(size == 16)
    {
        ulong64 &ul1 = *reinterpret_cast<ulong64 *>(this);
        ulong64 &ul2 = *(reinterpret_cast<ulong64 *>(this) + 1);
        return ul1 == static_cast<ulong64>(0) && ul2 == static_cast<ulong64>(0);
    }

    /// <summary>
    /// Useful helper method when using masks
    /// </summary>
    bool DEVICE_CODE AreAllFalse()
        requires(size == 32)
    {
        ulong64 *ptr = reinterpret_cast<ulong64 *>(this);
        ulong64 &ul1 = *ptr;
        ptr++;
        ulong64 &ul2 = *ptr;
        ptr++;
        ulong64 &ul3 = *ptr;
        ptr++;
        ulong64 &ul4 = *ptr;
        return ul1 == static_cast<ulong64>(0) && ul2 == static_cast<ulong64>(0) && ul3 == static_cast<ulong64>(0) &&
               ul4 == static_cast<ulong64>(0);
    }

    /// <summary>
    /// Useful helper method when using masks (and use SIMD for these large ones?)
    /// </summary>
    bool DEVICE_CODE AreAllFalse()
        requires(size == 64)
    {
        ulong64 *ptr = reinterpret_cast<ulong64 *>(this);
        ulong64 &ul1 = *ptr;
        ptr++;
        ulong64 &ul2 = *ptr;
        ptr++;
        ulong64 &ul3 = *ptr;
        ptr++;
        ulong64 &ul4 = *ptr;
        ptr++;
        ulong64 &ul5 = *ptr;
        ptr++;
        ulong64 &ul6 = *ptr;
        ptr++;
        ulong64 &ul7 = *ptr;
        ptr++;
        ulong64 &ul8 = *ptr;
        return ul1 == static_cast<ulong64>(0) && ul2 == static_cast<ulong64>(0) && ul3 == static_cast<ulong64>(0) &&
               ul4 == static_cast<ulong64>(0) && ul5 == static_cast<ulong64>(0) && ul6 == static_cast<ulong64>(0) &&
               ul7 == static_cast<ulong64>(0) && ul8 == static_cast<ulong64>(0);
    }

    /// <summary>
    /// Useful helper method when using masks, true means anything but 0
    /// </summary>
    bool DEVICE_CODE AreAllTrue()
        requires(size == 1)
    {
        return Tupel<byte, size>::value[0] != static_cast<byte>(0);
    }

    /// <summary>
    /// Useful helper method when using masks, true means anything but 0
    /// </summary>
    bool DEVICE_CODE AreAllTrue()
        requires(size == 2)
    {
        // two logical comparison are likely faster than any play with intrinsics...
        return Tupel<byte, size>::value[0] != static_cast<byte>(0) &&
               Tupel<byte, size>::value[1] != static_cast<byte>(0);
    }

    /// <summary>
    /// Useful helper method when using masks, true means anything but 0,
    /// default implementation
    /// </summary>
    bool DEVICE_CODE AreAllTrue()
        requires(size == 4)
    {
        return Tupel<byte, size>::value[0] != static_cast<byte>(0) &&
               Tupel<byte, size>::value[1] != static_cast<byte>(0) &&
               Tupel<byte, size>::value[2] != static_cast<byte>(0) &&
               Tupel<byte, size>::value[3] != static_cast<byte>(0);
    }

    /// <summary>
    /// Useful helper method when using masks, true means anything but 0,
    /// using CUDA intrinsics
    /// </summary>
    bool DEVICE_ONLY_CODE AreAllTrue()
        requires(size == 4) && //
                CUDA_ONLY<byte> && DeviceCode<byte>
    {
        uint &ui = *reinterpret_cast<uint *>(this);
        uint res = __vaddus4(ui, 0xFEFEFEFE); // add 254 to each byte with saturation
        // if the result is then 255 or FF for all bytes, none of them was 0 or false
        return res == static_cast<uint>(0xFFFFFFFF);
    }

    /// <summary>
    /// Useful helper method when using masks, true means anything but 0,
    /// using CUDA intrinsics
    /// </summary>
    bool DEVICE_ONLY_CODE AreAllTrue()
        requires(size == 8) && //
                CUDA_ONLY<byte>
    {
        uint &ui  = *reinterpret_cast<uint *>(this);
        uint res1 = __vaddus4(ui, 0xFEFEFEFE); // add 254 to each byte with saturation

        uint &ui2 = *(reinterpret_cast<uint *>(this) + 1);
        uint res2 = __vaddus4(ui2, 0xFEFEFEFE); // add 254 to each byte with saturation

        // if the result is then 255 or FF for all bytes, none of them was 0 or false
        return res1 == static_cast<uint>(0xFFFFFFFF) && res2 == static_cast<uint>(0xFFFFFFFF);
    }

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

    /// <summary>
    /// Useful helper method when using masks, true means anything but 0,
    /// default implementation
    /// </summary>
    bool DEVICE_CODE AreAllTrue()
        requires(size >= 8)
    {
        bool res = Tupel<byte, size>::value[0] > 0;
#pragma unroll
        for (size_t i = 1; i < size; i++)
        {
            res &= Tupel<byte, size>::value[i] > 0;
        }
        return res;
    }

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsEnd.h>
};
} // namespace opp