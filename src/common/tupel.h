#pragma once
#include <common/defines.h>
#include <concepts>
#include <cstddef>

namespace opp
{
// 64 8-bit values for 512 AVX instructions are current reasonable maximum
constexpr std::size_t MAX_TUPEL_SIZE = 64;

// size is Power of 2 and <= 64
template <std::size_t size>
concept IsTupelSize = IsPowerOf2<size> && size <= MAX_TUPEL_SIZE;

/// <summary>
/// A Tupel represents multiple elements that are aligned to the total tupel size.
/// Provides Load and Store functions for aligned and unaligned pointers.
/// </summary>
/// <typeparam name="T">The base type</typeparam>
/// <typeparam name="size">Number of elements, must be &lt;= 64 for Load and Store functions</typeparam>
template <typename T, std::size_t size> struct alignas(size * sizeof(T)) Tupel
{
    T value[size];

    /// <summary>
    /// Checks if the provided pointer is aligned to the Tupel size
    /// </summary>
    static bool DEVICE_CODE IsAligned(const T *aPtr)
        requires IsTupelSize<size>
    {
        return std::size_t(aPtr) % sizeof(Tupel<T, size>) == 0;
    }

    /// <summary>
    /// Loads Tupel size elements from aSrc. If aSrc is aligned to Tupel size, it performs one single load operation.
    /// Otherwise every single element is loaded individually.
    /// </summary>
    static Tupel<T, size> DEVICE_CODE Load(const T *aSrc)
        requires IsTupelSize<size>
    {
        if (IsAligned(aSrc))
        {
            return LoadAligned(aSrc);
        }

        return LoadUnaligned(aSrc);
    }

    /// <summary>
    /// Loads Tupel size elements from aSrc. Assumes that aSrc is aligned to Tupel size and doesn't perfrom any checks.
    /// </summary>
    static Tupel<T, size> DEVICE_CODE LoadAligned(const T *aSrc)
        requires IsTupelSize<size>
    {
        return *reinterpret_cast<const Tupel<T, size> *>(aSrc);
    }

    /// <summary>
    /// Loads Tupel size elements from aSrc. aSrc is assumed to be unaligned to Tupel size, it performs a load operation
    /// for every single element.
    /// </summary>
    static Tupel<T, size> DEVICE_CODE LoadUnaligned(const T *aSrc)
        requires IsTupelSize<size>
    {
        Tupel<T, size> ret;

        ret.value[0] = *aSrc;
        if constexpr (size > 1)
        {
            aSrc++;
            ret.value[1] = *aSrc;
        }
        if constexpr (size > 2)
        {
            aSrc++;
            ret.value[2] = *aSrc;
        }
        if constexpr (size > 3)
        {
            aSrc++;
            ret.value[3] = *aSrc;
        }
        if constexpr (size > 4)
        {
            aSrc++;
            ret.value[4] = *aSrc;
        }
        if constexpr (size > 5)
        {
            aSrc++;
            ret.value[5] = *aSrc;
        }
        if constexpr (size > 6)
        {
            aSrc++;
            ret.value[6] = *aSrc;
        }
        if constexpr (size > 7)
        {
            aSrc++;
            ret.value[7] = *aSrc;
        }
        if constexpr (size > 8)
        {
            aSrc++;
            ret.value[8] = *aSrc;
        }
        if constexpr (size > 9)
        {
            aSrc++;
            ret.value[9] = *aSrc;
        }
        if constexpr (size > 10)
        {
            aSrc++;
            ret.value[10] = *aSrc;
        }
        if constexpr (size > 11)
        {
            aSrc++;
            ret.value[11] = *aSrc;
        }
        if constexpr (size > 12)
        {
            aSrc++;
            ret.value[12] = *aSrc;
        }
        if constexpr (size > 13)
        {
            aSrc++;
            ret.value[13] = *aSrc;
        }
        if constexpr (size > 14)
        {
            aSrc++;
            ret.value[14] = *aSrc;
        }
        if constexpr (size > 15)
        {
            aSrc++;
            ret.value[15] = *aSrc;
        }
        if constexpr (size > 16)
        {
            aSrc++;
            ret.value[16] = *aSrc;
        }
        if constexpr (size > 17)
        {
            aSrc++;
            ret.value[17] = *aSrc;
        }
        if constexpr (size > 18)
        {
            aSrc++;
            ret.value[18] = *aSrc;
        }
        if constexpr (size > 19)
        {
            aSrc++;
            ret.value[19] = *aSrc;
        }
        if constexpr (size > 20)
        {
            aSrc++;
            ret.value[20] = *aSrc;
        }
        if constexpr (size > 21)
        {
            aSrc++;
            ret.value[21] = *aSrc;
        }
        if constexpr (size > 22)
        {
            aSrc++;
            ret.value[22] = *aSrc;
        }
        if constexpr (size > 23)
        {
            aSrc++;
            ret.value[23] = *aSrc;
        }
        if constexpr (size > 24)
        {
            aSrc++;
            ret.value[24] = *aSrc;
        }
        if constexpr (size > 25)
        {
            aSrc++;
            ret.value[25] = *aSrc;
        }
        if constexpr (size > 26)
        {
            aSrc++;
            ret.value[26] = *aSrc;
        }
        if constexpr (size > 27)
        {
            aSrc++;
            ret.value[27] = *aSrc;
        }
        if constexpr (size > 28)
        {
            aSrc++;
            ret.value[28] = *aSrc;
        }
        if constexpr (size > 29)
        {
            aSrc++;
            ret.value[29] = *aSrc;
        }
        if constexpr (size > 30)
        {
            aSrc++;
            ret.value[30] = *aSrc;
        }
        if constexpr (size > 31)
        {
            aSrc++;
            ret.value[31] = *aSrc;
        }
        if constexpr (size > 32)
        {
            aSrc++;
            ret.value[32] = *aSrc;
        }
        if constexpr (size > 33)
        {
            aSrc++;
            ret.value[33] = *aSrc;
        }
        if constexpr (size > 34)
        {
            aSrc++;
            ret.value[34] = *aSrc;
        }
        if constexpr (size > 35)
        {
            aSrc++;
            ret.value[35] = *aSrc;
        }
        if constexpr (size > 36)
        {
            aSrc++;
            ret.value[36] = *aSrc;
        }
        if constexpr (size > 37)
        {
            aSrc++;
            ret.value[37] = *aSrc;
        }
        if constexpr (size > 38)
        {
            aSrc++;
            ret.value[38] = *aSrc;
        }
        if constexpr (size > 39)
        {
            aSrc++;
            ret.value[39] = *aSrc;
        }
        if constexpr (size > 40)
        {
            aSrc++;
            ret.value[40] = *aSrc;
        }
        if constexpr (size > 41)
        {
            aSrc++;
            ret.value[41] = *aSrc;
        }
        if constexpr (size > 42)
        {
            aSrc++;
            ret.value[42] = *aSrc;
        }
        if constexpr (size > 43)
        {
            aSrc++;
            ret.value[43] = *aSrc;
        }
        if constexpr (size > 44)
        {
            aSrc++;
            ret.value[44] = *aSrc;
        }
        if constexpr (size > 45)
        {
            aSrc++;
            ret.value[45] = *aSrc;
        }
        if constexpr (size > 46)
        {
            aSrc++;
            ret.value[46] = *aSrc;
        }
        if constexpr (size > 47)
        {
            aSrc++;
            ret.value[47] = *aSrc;
        }
        if constexpr (size > 48)
        {
            aSrc++;
            ret.value[48] = *aSrc;
        }
        if constexpr (size > 49)
        {
            aSrc++;
            ret.value[49] = *aSrc;
        }
        if constexpr (size > 50)
        {
            aSrc++;
            ret.value[50] = *aSrc;
        }
        if constexpr (size > 51)
        {
            aSrc++;
            ret.value[51] = *aSrc;
        }
        if constexpr (size > 52)
        {
            aSrc++;
            ret.value[52] = *aSrc;
        }
        if constexpr (size > 53)
        {
            aSrc++;
            ret.value[53] = *aSrc;
        }
        if constexpr (size > 54)
        {
            aSrc++;
            ret.value[54] = *aSrc;
        }
        if constexpr (size > 55)
        {
            aSrc++;
            ret.value[55] = *aSrc;
        }
        if constexpr (size > 56)
        {
            aSrc++;
            ret.value[56] = *aSrc;
        }
        if constexpr (size > 57)
        {
            aSrc++;
            ret.value[57] = *aSrc;
        }
        if constexpr (size > 58)
        {
            aSrc++;
            ret.value[58] = *aSrc;
        }
        if constexpr (size > 59)
        {
            aSrc++;
            ret.value[59] = *aSrc;
        }
        if constexpr (size > 60)
        {
            aSrc++;
            ret.value[60] = *aSrc;
        }
        if constexpr (size > 61)
        {
            aSrc++;
            ret.value[61] = *aSrc;
        }
        if constexpr (size > 62)
        {
            aSrc++;
            ret.value[62] = *aSrc;
        }
        if constexpr (size > 63)
        {
            aSrc++;
            ret.value[63] = *aSrc;
        }
        if constexpr (size > MAX_TUPEL_SIZE)
        {
            // shouldn't happen but let's stay on the safe side...
            static_assert(AlwaysFalse<T>::value, "Maximum Tupel size is 64.");
        }

        return ret;
    }

    /// <summary>
    /// Stores Tupel size elements to aDst. If aDst is aligned to Tupel size, it performs one single store operation.
    /// Otherwise every single element is stored individually.
    /// </summary>
    static void DEVICE_CODE Store(const Tupel<T, size> &aSrc, T *aDst)
        requires IsTupelSize<size>
    {
        if (IsAligned(aDst))
        {
            StoreAligned(aSrc, aDst);
            return;
        }

        StoreUnaligned(aSrc, aDst);
    }

    /// <summary>
    /// Stores Tupel size elements to aDst. Assumes that aDst is aligned to Tupel size and doesn't perfrom any checks.
    /// </summary>
    static void DEVICE_CODE StoreAligned(const Tupel<T, size> &aSrc, T *aDst)
        requires IsTupelSize<size>
    {
        *reinterpret_cast<Tupel<T, size> *>(aDst) = aSrc;
    }

    /// <summary>
    /// Stores Tupel size elements to aDst. aDst is assumed to be unaligned to Tupel size, it performs a store operation
    /// for every single element.
    /// </summary>
    static void DEVICE_CODE StoreUnaligned(const Tupel<T, size> &aSrc, T *aDst)
        requires IsTupelSize<size>
    {
        *aDst = aSrc.value[0];
        if constexpr (size > 1)
        {
            aDst++;
            *aDst = aSrc.value[1];
        }
        if constexpr (size > 2)
        {
            aDst++;
            *aDst = aSrc.value[2];
        }
        if constexpr (size > 3)
        {
            aDst++;
            *aDst = aSrc.value[3];
        }
        if constexpr (size > 4)
        {
            aDst++;
            *aDst = aSrc.value[4];
        }
        if constexpr (size > 5)
        {
            aDst++;
            *aDst = aSrc.value[5];
        }
        if constexpr (size > 6)
        {
            aDst++;
            *aDst = aSrc.value[6];
        }
        if constexpr (size > 7)
        {
            aDst++;
            *aDst = aSrc.value[7];
        }
        if constexpr (size > 8)
        {
            aDst++;
            *aDst = aSrc.value[8];
        }
        if constexpr (size > 9)
        {
            aDst++;
            *aDst = aSrc.value[9];
        }
        if constexpr (size > 10)
        {
            aDst++;
            *aDst = aSrc.value[10];
        }
        if constexpr (size > 11)
        {
            aDst++;
            *aDst = aSrc.value[11];
        }
        if constexpr (size > 12)
        {
            aDst++;
            *aDst = aSrc.value[12];
        }
        if constexpr (size > 13)
        {
            aDst++;
            *aDst = aSrc.value[13];
        }
        if constexpr (size > 14)
        {
            aDst++;
            *aDst = aSrc.value[14];
        }
        if constexpr (size > 15)
        {
            aDst++;
            *aDst = aSrc.value[15];
        }
        if constexpr (size > 16)
        {
            aDst++;
            *aDst = aSrc.value[16];
        }
        if constexpr (size > 17)
        {
            aDst++;
            *aDst = aSrc.value[17];
        }
        if constexpr (size > 18)
        {
            aDst++;
            *aDst = aSrc.value[18];
        }
        if constexpr (size > 19)
        {
            aDst++;
            *aDst = aSrc.value[19];
        }
        if constexpr (size > 20)
        {
            aDst++;
            *aDst = aSrc.value[20];
        }
        if constexpr (size > 21)
        {
            aDst++;
            *aDst = aSrc.value[21];
        }
        if constexpr (size > 22)
        {
            aDst++;
            *aDst = aSrc.value[22];
        }
        if constexpr (size > 23)
        {
            aDst++;
            *aDst = aSrc.value[23];
        }
        if constexpr (size > 24)
        {
            aDst++;
            *aDst = aSrc.value[24];
        }
        if constexpr (size > 25)
        {
            aDst++;
            *aDst = aSrc.value[25];
        }
        if constexpr (size > 26)
        {
            aDst++;
            *aDst = aSrc.value[26];
        }
        if constexpr (size > 27)
        {
            aDst++;
            *aDst = aSrc.value[27];
        }
        if constexpr (size > 28)
        {
            aDst++;
            *aDst = aSrc.value[28];
        }
        if constexpr (size > 29)
        {
            aDst++;
            *aDst = aSrc.value[29];
        }
        if constexpr (size > 30)
        {
            aDst++;
            *aDst = aSrc.value[30];
        }
        if constexpr (size > 31)
        {
            aDst++;
            *aDst = aSrc.value[31];
        }
        if constexpr (size > 32)
        {
            aDst++;
            *aDst = aSrc.value[32];
        }
        if constexpr (size > 33)
        {
            aDst++;
            *aDst = aSrc.value[33];
        }
        if constexpr (size > 34)
        {
            aDst++;
            *aDst = aSrc.value[34];
        }
        if constexpr (size > 35)
        {
            aDst++;
            *aDst = aSrc.value[35];
        }
        if constexpr (size > 36)
        {
            aDst++;
            *aDst = aSrc.value[36];
        }
        if constexpr (size > 37)
        {
            aDst++;
            *aDst = aSrc.value[37];
        }
        if constexpr (size > 38)
        {
            aDst++;
            *aDst = aSrc.value[38];
        }
        if constexpr (size > 39)
        {
            aDst++;
            *aDst = aSrc.value[39];
        }
        if constexpr (size > 40)
        {
            aDst++;
            *aDst = aSrc.value[40];
        }
        if constexpr (size > 41)
        {
            aDst++;
            *aDst = aSrc.value[41];
        }
        if constexpr (size > 42)
        {
            aDst++;
            *aDst = aSrc.value[42];
        }
        if constexpr (size > 43)
        {
            aDst++;
            *aDst = aSrc.value[43];
        }
        if constexpr (size > 44)
        {
            aDst++;
            *aDst = aSrc.value[44];
        }
        if constexpr (size > 45)
        {
            aDst++;
            *aDst = aSrc.value[45];
        }
        if constexpr (size > 46)
        {
            aDst++;
            *aDst = aSrc.value[46];
        }
        if constexpr (size > 47)
        {
            aDst++;
            *aDst = aSrc.value[47];
        }
        if constexpr (size > 48)
        {
            aDst++;
            *aDst = aSrc.value[48];
        }
        if constexpr (size > 49)
        {
            aDst++;
            *aDst = aSrc.value[49];
        }
        if constexpr (size > 50)
        {
            aDst++;
            *aDst = aSrc.value[50];
        }
        if constexpr (size > 51)
        {
            aDst++;
            *aDst = aSrc.value[51];
        }
        if constexpr (size > 52)
        {
            aDst++;
            *aDst = aSrc.value[52];
        }
        if constexpr (size > 53)
        {
            aDst++;
            *aDst = aSrc.value[53];
        }
        if constexpr (size > 54)
        {
            aDst++;
            *aDst = aSrc.value[54];
        }
        if constexpr (size > 55)
        {
            aDst++;
            *aDst = aSrc.value[55];
        }
        if constexpr (size > 56)
        {
            aDst++;
            *aDst = aSrc.value[56];
        }
        if constexpr (size > 57)
        {
            aDst++;
            *aDst = aSrc.value[57];
        }
        if constexpr (size > 58)
        {
            aDst++;
            *aDst = aSrc.value[58];
        }
        if constexpr (size > 59)
        {
            aDst++;
            *aDst = aSrc.value[59];
        }
        if constexpr (size > 60)
        {
            aDst++;
            *aDst = aSrc.value[60];
        }
        if constexpr (size > 61)
        {
            aDst++;
            *aDst = aSrc.value[61];
        }
        if constexpr (size > 62)
        {
            aDst++;
            *aDst = aSrc.value[62];
        }
        if constexpr (size > 63)
        {
            aDst++;
            *aDst = aSrc.value[63];
        }
        if constexpr (size > MAX_TUPEL_SIZE)
        {
            // shouldn't happen but let's stay on the safe side...
            static_assert(AlwaysFalse<T>::value, "Maximum Tupel size is 64.");
        }
    }
};
} // namespace opp