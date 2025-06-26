#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/maskTupel.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace mpp;
using namespace Catch;

TEST_CASE("Tupel size 2", "[Common]")
{
    constexpr size_t bufferSize = 1024;
    constexpr size_t tupelSize  = 2;
    std::vector<int> buffer(bufferSize);
    std::iota(buffer.begin(), buffer.end(), 0);

    int *alignedPtr   = nullptr;
    int *unalignedPtr = nullptr;
    size_t firstElem  = 0;

    for (size_t i = 0; i < bufferSize - 1; i++)
    {
        if (size_t(&buffer[i]) % (tupelSize * sizeof(int)) == 0)
        {
            alignedPtr   = &buffer[i];
            unalignedPtr = &buffer[i + 1];
            firstElem    = i;
            break;
        }
    }

    CHECK(Tupel<int, tupelSize>::IsAligned(alignedPtr) == true);
    CHECK(Tupel<int, tupelSize>::IsAligned(unalignedPtr) == false);

    Tupel<int, tupelSize> tupelAligned   = Tupel<int, tupelSize>::Load(alignedPtr);
    Tupel<int, tupelSize> tupelUnaligned = Tupel<int, tupelSize>::Load(unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(tupelAligned.value[i] == to_int(firstElem + i));
        CHECK(tupelUnaligned.value[i] == to_int(firstElem + i + 1));
    }

    for (size_t i = 0; i < tupelSize; i++)
    {
        tupelAligned.value[i]   = to_int(100 + i);
        tupelUnaligned.value[i] = to_int(200 + i);
    }

    Tupel<int, tupelSize>::Store(tupelAligned, alignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i] == to_int(100 + i));
    }

    Tupel<int, tupelSize>::Store(tupelUnaligned, unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i + 1] == to_int(200 + i));
    }
}

TEST_CASE("Tupel size 4", "[Common]")
{
    constexpr size_t bufferSize = 1024;
    constexpr size_t tupelSize  = 4;
    std::vector<int> buffer(bufferSize);
    std::iota(buffer.begin(), buffer.end(), 0);

    int *alignedPtr   = nullptr;
    int *unalignedPtr = nullptr;
    size_t firstElem  = 0;

    for (size_t i = 0; i < bufferSize - 1; i++)
    {
        if (size_t(&buffer[i]) % (tupelSize * sizeof(int)) == 0)
        {
            alignedPtr   = &buffer[i];
            unalignedPtr = &buffer[i + 1];
            firstElem    = i;
            break;
        }
    }

    CHECK(Tupel<int, tupelSize>::IsAligned(alignedPtr) == true);
    CHECK(Tupel<int, tupelSize>::IsAligned(unalignedPtr) == false);

    Tupel<int, tupelSize> tupelAligned   = Tupel<int, tupelSize>::Load(alignedPtr);
    Tupel<int, tupelSize> tupelUnaligned = Tupel<int, tupelSize>::Load(unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(tupelAligned.value[i] == to_int(firstElem + i));
        CHECK(tupelUnaligned.value[i] == to_int(firstElem + i + 1));
    }

    for (size_t i = 0; i < tupelSize; i++)
    {
        tupelAligned.value[i]   = to_int(100 + i);
        tupelUnaligned.value[i] = to_int(200 + i);
    }

    Tupel<int, tupelSize>::Store(tupelAligned, alignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i] == to_int(100 + i));
    }

    Tupel<int, tupelSize>::Store(tupelUnaligned, unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i + 1] == to_int(200 + i));
    }
}

TEST_CASE("Tupel size 8", "[Common]")
{
    constexpr size_t bufferSize = 1024;
    constexpr size_t tupelSize  = 8;
    std::vector<int> buffer(bufferSize);
    std::iota(buffer.begin(), buffer.end(), 0);

    int *alignedPtr   = nullptr;
    int *unalignedPtr = nullptr;
    size_t firstElem  = 0;

    for (size_t i = 0; i < bufferSize - 1; i++)
    {
        if (size_t(&buffer[i]) % (tupelSize * sizeof(int)) == 0)
        {
            alignedPtr   = &buffer[i];
            unalignedPtr = &buffer[i + 1];
            firstElem    = i;
            break;
        }
    }

    CHECK(Tupel<int, tupelSize>::IsAligned(alignedPtr) == true);
    CHECK(Tupel<int, tupelSize>::IsAligned(unalignedPtr) == false);

    Tupel<int, tupelSize> tupelAligned   = Tupel<int, tupelSize>::Load(alignedPtr);
    Tupel<int, tupelSize> tupelUnaligned = Tupel<int, tupelSize>::Load(unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(tupelAligned.value[i] == to_int(firstElem + i));
        CHECK(tupelUnaligned.value[i] == to_int(firstElem + i + 1));
    }

    for (size_t i = 0; i < tupelSize; i++)
    {
        tupelAligned.value[i]   = to_int(100 + i);
        tupelUnaligned.value[i] = to_int(200 + i);
    }

    Tupel<int, tupelSize>::Store(tupelAligned, alignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i] == to_int(100 + i));
    }

    Tupel<int, tupelSize>::Store(tupelUnaligned, unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i + 1] == to_int(200 + i));
    }
}

TEST_CASE("Tupel size 16", "[Common]")
{
    constexpr size_t bufferSize = 1024;
    constexpr size_t tupelSize  = 16;
    std::vector<int> buffer(bufferSize);
    std::iota(buffer.begin(), buffer.end(), 0);

    int *alignedPtr   = nullptr;
    int *unalignedPtr = nullptr;
    size_t firstElem  = 0;

    for (size_t i = 0; i < bufferSize - 1; i++)
    {
        if (size_t(&buffer[i]) % (tupelSize * sizeof(int)) == 0)
        {
            alignedPtr   = &buffer[i];
            unalignedPtr = &buffer[i + 1];
            firstElem    = i;
            break;
        }
    }

    CHECK(Tupel<int, tupelSize>::IsAligned(alignedPtr) == true);
    CHECK(Tupel<int, tupelSize>::IsAligned(unalignedPtr) == false);

    Tupel<int, tupelSize> tupelAligned   = Tupel<int, tupelSize>::Load(alignedPtr);
    Tupel<int, tupelSize> tupelUnaligned = Tupel<int, tupelSize>::Load(unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(tupelAligned.value[i] == to_int(firstElem + i));
        CHECK(tupelUnaligned.value[i] == to_int(firstElem + i + 1));
    }

    for (size_t i = 0; i < tupelSize; i++)
    {
        tupelAligned.value[i]   = to_int(100 + i);
        tupelUnaligned.value[i] = to_int(200 + i);
    }

    Tupel<int, tupelSize>::Store(tupelAligned, alignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i] == to_int(100 + i));
    }

    Tupel<int, tupelSize>::Store(tupelUnaligned, unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i + 1] == to_int(200 + i));
    }
}

TEST_CASE("Tupel size 32", "[Common]")
{
    constexpr size_t bufferSize = 1024;
    constexpr size_t tupelSize  = 32;
    std::vector<int> buffer(bufferSize);
    std::iota(buffer.begin(), buffer.end(), 0);

    int *alignedPtr   = nullptr;
    int *unalignedPtr = nullptr;
    size_t firstElem  = 0;

    for (size_t i = 0; i < bufferSize - 1; i++)
    {
        if (size_t(&buffer[i]) % (tupelSize * sizeof(int)) == 0)
        {
            alignedPtr   = &buffer[i];
            unalignedPtr = &buffer[i + 1];
            firstElem    = i;
            break;
        }
    }

    CHECK(Tupel<int, tupelSize>::IsAligned(alignedPtr) == true);
    CHECK(Tupel<int, tupelSize>::IsAligned(unalignedPtr) == false);

    Tupel<int, tupelSize> tupelAligned   = Tupel<int, tupelSize>::Load(alignedPtr);
    Tupel<int, tupelSize> tupelUnaligned = Tupel<int, tupelSize>::Load(unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(tupelAligned.value[i] == to_int(firstElem + i));
        CHECK(tupelUnaligned.value[i] == to_int(firstElem + i + 1));
    }

    for (size_t i = 0; i < tupelSize; i++)
    {
        tupelAligned.value[i]   = to_int(100 + i);
        tupelUnaligned.value[i] = to_int(200 + i);
    }

    Tupel<int, tupelSize>::Store(tupelAligned, alignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i] == to_int(100 + i));
    }

    Tupel<int, tupelSize>::Store(tupelUnaligned, unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i + 1] == to_int(200 + i));
    }
}

TEST_CASE("Tupel size 64", "[Common]")
{
    constexpr size_t bufferSize = 1024;
    constexpr size_t tupelSize  = 64;
    std::vector<int> buffer(bufferSize);
    std::iota(buffer.begin(), buffer.end(), 0);

    int *alignedPtr   = nullptr;
    int *unalignedPtr = nullptr;
    size_t firstElem  = 0;

    for (size_t i = 0; i < bufferSize - 1; i++)
    {
        if (size_t(&buffer[i]) % (tupelSize * sizeof(int)) == 0)
        {
            alignedPtr   = &buffer[i];
            unalignedPtr = &buffer[i + 1];
            firstElem    = i;
            break;
        }
    }

    CHECK(Tupel<int, tupelSize>::IsAligned(alignedPtr) == true);
    CHECK(Tupel<int, tupelSize>::IsAligned(unalignedPtr) == false);

    Tupel<int, tupelSize> tupelAligned   = Tupel<int, tupelSize>::Load(alignedPtr);
    Tupel<int, tupelSize> tupelUnaligned = Tupel<int, tupelSize>::Load(unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(tupelAligned.value[i] == to_int(firstElem + i));
        CHECK(tupelUnaligned.value[i] == to_int(firstElem + i + 1));
    }

    for (size_t i = 0; i < tupelSize; i++)
    {
        tupelAligned.value[i]   = to_int(100 + i);
        tupelUnaligned.value[i] = to_int(200 + i);
    }

    Tupel<int, tupelSize>::Store(tupelAligned, alignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i] == to_int(100 + i));
    }

    Tupel<int, tupelSize>::Store(tupelUnaligned, unalignedPtr);

    for (size_t i = 0; i < tupelSize; i++)
    {
        CHECK(buffer[firstElem + i + 1] == to_int(200 + i));
    }
}

TEST_CASE("MaskTupel size 2", "[Common]")
{
    constexpr size_t tupelSize = 2;

    MaskTupel<tupelSize> mask;

    for (int trueValue = 1; trueValue < 256; trueValue++)
    {
        memset(mask.value, trueValue, tupelSize);

        CHECK(mask.AreAllFalse() == false);
        CHECK(mask.AreAllTrue() == true);

        memset(mask.value, 0, tupelSize);

        CHECK(mask.AreAllFalse() == true);
        CHECK(mask.AreAllTrue() == false);

        for (size_t i = 0; i < tupelSize; i++)
        {
            mask.value[i] = to_byte(trueValue);

            CHECK(mask.AreAllFalse() == false);
            CHECK(mask.AreAllTrue() == false);

            mask.value[i] = 0;
        }
    }
}

TEST_CASE("MaskTupel size 4", "[Common]")
{
    constexpr size_t tupelSize = 4;

    MaskTupel<tupelSize> mask;

    for (int trueValue = 1; trueValue < 256; trueValue++)
    {
        memset(mask.value, trueValue, tupelSize);

        CHECK(mask.AreAllFalse() == false);
        CHECK(mask.AreAllTrue() == true);

        memset(mask.value, 0, tupelSize);

        CHECK(mask.AreAllFalse() == true);
        CHECK(mask.AreAllTrue() == false);

        for (size_t i = 0; i < tupelSize; i++)
        {
            mask.value[i] = to_byte(trueValue);

            CHECK(mask.AreAllFalse() == false);
            CHECK(mask.AreAllTrue() == false);

            mask.value[i] = 0;
        }
    }
}

TEST_CASE("MaskTupel size 8", "[Common]")
{
    constexpr size_t tupelSize = 8;

    MaskTupel<tupelSize> mask;

    for (int trueValue = 1; trueValue < 256; trueValue++)
    {
        memset(mask.value, trueValue, tupelSize);

        CHECK(mask.AreAllFalse() == false);
        CHECK(mask.AreAllTrue() == true);

        memset(mask.value, 0, tupelSize);

        CHECK(mask.AreAllFalse() == true);
        CHECK(mask.AreAllTrue() == false);

        for (size_t i = 0; i < tupelSize; i++)
        {
            mask.value[i] = to_byte(trueValue);

            CHECK(mask.AreAllFalse() == false);
            CHECK(mask.AreAllTrue() == false);

            mask.value[i] = 0;
        }
    }
}

TEST_CASE("MaskTupel size 16", "[Common]")
{
    constexpr size_t tupelSize = 16;

    MaskTupel<tupelSize> mask;

    for (int trueValue = 1; trueValue < 256; trueValue++)
    {
        memset(mask.value, trueValue, tupelSize);

        CHECK(mask.AreAllFalse() == false);
        CHECK(mask.AreAllTrue() == true);

        memset(mask.value, 0, tupelSize);

        CHECK(mask.AreAllFalse() == true);
        CHECK(mask.AreAllTrue() == false);

        for (size_t i = 0; i < tupelSize; i++)
        {
            mask.value[i] = to_byte(trueValue);

            CHECK(mask.AreAllFalse() == false);
            CHECK(mask.AreAllTrue() == false);

            mask.value[i] = 0;
        }
    }
}

TEST_CASE("MaskTupel size 32", "[Common]")
{
    constexpr size_t tupelSize = 32;

    MaskTupel<tupelSize> mask;

    for (int trueValue = 1; trueValue < 256; trueValue++)
    {
        memset(mask.value, trueValue, tupelSize);

        CHECK(mask.AreAllFalse() == false);
        CHECK(mask.AreAllTrue() == true);

        memset(mask.value, 0, tupelSize);

        CHECK(mask.AreAllFalse() == true);
        CHECK(mask.AreAllTrue() == false);

        for (size_t i = 0; i < tupelSize; i++)
        {
            mask.value[i] = to_byte(trueValue);

            CHECK(mask.AreAllFalse() == false);
            CHECK(mask.AreAllTrue() == false);

            mask.value[i] = 0;
        }
    }
}

TEST_CASE("MaskTupel size 64", "[Common]")
{
    constexpr size_t tupelSize = 64;

    MaskTupel<tupelSize> mask;

    for (int trueValue = 1; trueValue < 256; trueValue++)
    {
        memset(mask.value, trueValue, tupelSize);

        CHECK(mask.AreAllFalse() == false);
        CHECK(mask.AreAllTrue() == true);

        memset(mask.value, 0, tupelSize);

        CHECK(mask.AreAllFalse() == true);
        CHECK(mask.AreAllTrue() == false);

        for (size_t i = 0; i < tupelSize; i++)
        {
            mask.value[i] = to_byte(trueValue);

            CHECK(mask.AreAllFalse() == false);
            CHECK(mask.AreAllTrue() == false);

            mask.value[i] = 0;
        }
    }
}
