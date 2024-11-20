#pragma once
#include <common/defines.h>
#include <cstddef>

namespace opp
{
/// <summary>
/// A Tupel represents multiple elements that are aligned to the total tupel size.
/// Provides Load and Store functions for aligned and unaligned pointers.
/// </summary>
/// <typeparam name="T">The base type</typeparam>
/// <typeparam name="size">Number of elements, must be &lt;= 16 for Load and Store functions</typeparam>
template <typename T, size_t size> struct alignas(size * sizeof(T)) Tupel
{
    T value[size];

    /// <summary>
    /// Checks if the provided pointer is aligned to the Tupel size
    /// </summary>
    static bool DEVICE_CODE IsAligned(const T *aPtr)
        requires IsTupelSize<size>
    {
        return size_t(aPtr) % sizeof(Tupel<T, size>) == 0;
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
            return *((Tupel<T, size> *)aSrc);
        }

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
            static_assert(AlwaysFalse<T>::value, "Maximum Tupel size is 16.");
        }

        return ret;
    }

    /// <summary>
    /// Loads Tupel size elements from aSrc. Assumes that aSrc is aligned to Tupel size and doesn't perfrom any checks.
    /// </summary>
    static Tupel<T, size> DEVICE_CODE LoadAligned(const T *aSrc)
        requires IsTupelSize<size>
    {
        return *((Tupel<T, size> *)aSrc);
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
            static_assert(AlwaysFalse<T>::value, "Maximum Tupel size is 16.");
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
            *((Tupel<T, size> *)aDst) = aSrc;
            return;
        }

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
            static_assert(AlwaysFalse<T>::value, "Maximum Tupel size is 16.");
        }
    }

    /// <summary>
    /// Stores Tupel size elements to aDst. Assumes that aDst is aligned to Tupel size and doesn't perfrom any checks.
    /// </summary>
    static void DEVICE_CODE StoreAligned(const Tupel<T, size> &aSrc, T *aDst)
        requires IsTupelSize<size>
    {
        *((Tupel<T, size> *)aDst) = aSrc;
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
            static_assert(AlwaysFalse<T>::value, "Maximum Tupel size is 16.");
        }
    }
};
} // namespace opp