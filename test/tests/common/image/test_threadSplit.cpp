#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/pixelTypes.h>
#include <common/image/threadSplit.h>
#include <common/safeCast.h>
#include <common/vector2.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace opp;
using namespace opp::image;
using namespace Catch;

// Pointers should all be allocated with a multiple of 256, the alignment that cuda_malloc gives us
void *address          = reinterpret_cast<void *>(8192);
const int bytesPerWarp = 256;
const int dataSize     = 4096;

TEST_CASE("ThreadSplit - 4 bytes base type", "[Common.Image]")
{
    Pixel32fC4 *vec32fC4 = reinterpret_cast<Pixel32fC4 *>(address);
    Pixel32fC3 *vec32fC3 = reinterpret_cast<Pixel32fC3 *>(address);
    Pixel32fC2 *vec32fC2 = reinterpret_cast<Pixel32fC2 *>(address);
    Pixel32fC1 *vec32fC1 = reinterpret_cast<Pixel32fC1 *>(address);

    // Vector3 has 12 bytes and can't be a multiple of bytesPerWarp, impossible to align
    CHECK_THROWS_AS((ThreadSplit<bytesPerWarp, 2>(vec32fC3, dataSize)), InvalidArgumentException);

    CHECK_NOTHROW(ThreadSplit<bytesPerWarp, 2>(vec32fC4, dataSize));
    CHECK_NOTHROW(ThreadSplit<bytesPerWarp, 2>(vec32fC1, dataSize));

    ThreadSplit<bytesPerWarp, 1> ts32fC4_Aligned(vec32fC4, dataSize);
    ThreadSplit<bytesPerWarp, 1> ts32fC4_Unaligned(vec32fC4 + 1, dataSize);

    ThreadSplit<bytesPerWarp, 1> ts32fC3_Aligned(vec32fC3, dataSize);
    ThreadSplit<bytesPerWarp, 1> ts32fC3_Unaligned(vec32fC3 + 1, dataSize);

    ThreadSplit<bytesPerWarp, 1> ts32fC2_Aligned(vec32fC2, dataSize);
    ThreadSplit<bytesPerWarp, 1> ts32fC2_Unaligned(vec32fC2 + 1, dataSize);

    ThreadSplit<bytesPerWarp, 2> ts32fC1_Aligned(vec32fC1, dataSize);
    ThreadSplit<bytesPerWarp, 2> ts32fC1_Unaligned(vec32fC1 + 1, dataSize);

    CHECK(ts32fC4_Aligned.Muted() == 0);
    CHECK(ts32fC4_Aligned.Left() == 0);
    CHECK(ts32fC4_Aligned.Center() == dataSize);
    CHECK(ts32fC4_Aligned.Right() == 0);

    CHECK(ts32fC4_Unaligned.Muted() == 0);
    CHECK(ts32fC4_Unaligned.Left() == 0);
    CHECK(ts32fC4_Unaligned.Center() == dataSize);
    CHECK(ts32fC4_Unaligned.Right() == 0);

    CHECK(ts32fC3_Aligned.Muted() == 0);
    CHECK(ts32fC3_Aligned.Left() == 0);
    CHECK(ts32fC3_Aligned.Center() == dataSize);
    CHECK(ts32fC3_Aligned.Right() == 0);

    CHECK(ts32fC3_Unaligned.Muted() == 0);
    CHECK(ts32fC3_Unaligned.Left() == 0);
    CHECK(ts32fC3_Unaligned.Center() == dataSize);
    CHECK(ts32fC3_Unaligned.Right() == 0);

    CHECK(ts32fC2_Aligned.Muted() == 0);
    CHECK(ts32fC2_Aligned.Left() == 0);
    CHECK(ts32fC2_Aligned.Center() == dataSize);
    CHECK(ts32fC2_Aligned.Right() == 0);

    CHECK(ts32fC2_Unaligned.Muted() == 0);
    CHECK(ts32fC2_Unaligned.Left() == 0);
    CHECK(ts32fC2_Unaligned.Center() == dataSize);
    CHECK(ts32fC2_Unaligned.Right() == 0);

    CHECK(ts32fC1_Aligned.Muted() == 0);
    CHECK(ts32fC1_Aligned.Left() == 0);
    CHECK(ts32fC1_Aligned.Center() == dataSize / 2);
    CHECK(ts32fC1_Aligned.Right() == 0);

    CHECK(ts32fC1_Unaligned.Muted() == 1);
    CHECK(ts32fC1_Unaligned.Left() == 63);
    CHECK(ts32fC1_Unaligned.Center() == (dataSize - (bytesPerWarp / to_int(sizeof(Pixel32fC1)))) / 2);
    CHECK(ts32fC1_Unaligned.Right() == 1);
    CHECK(ts32fC1_Unaligned.Left() * to_int(sizeof(Pixel32fC1)) < bytesPerWarp);
    CHECK(ts32fC1_Unaligned.Right() * to_int(sizeof(Pixel32fC1)) < bytesPerWarp);

    int pixelsC4Aligned             = 0;
    int threadsC4AlignedAligned     = 0;
    int threadsC4AlignedUnaligned   = 0;
    int pixelsC4Unaligned           = 0;
    int threadsC4UnalignedAligned   = 0;
    int threadsC4UnalignedUnaligned = 0;

    int pixelsC3Aligned             = 0;
    int threadsC3AlignedAligned     = 0;
    int threadsC3AlignedUnaligned   = 0;
    int pixelsC3Unaligned           = 0;
    int threadsC3UnalignedAligned   = 0;
    int threadsC3UnalignedUnaligned = 0;

    int pixelsC2Aligned             = 0;
    int threadsC2AlignedAligned     = 0;
    int threadsC2AlignedUnaligned   = 0;
    int pixelsC2Unaligned           = 0;
    int threadsC2UnalignedAligned   = 0;
    int threadsC2UnalignedUnaligned = 0;

    int pixelsC1Aligned             = 0;
    int threadsC1AlignedAligned     = 0;
    int threadsC1AlignedUnaligned   = 0;
    int pixelsC1Unaligned           = 0;
    int threadsC1UnalignedAligned   = 0;
    int threadsC1UnalignedUnaligned = 0;

    // i in this loop represents a processing thread (CUDA thread)
    // count the total number of processed pixels in aligned and unaligned splitting parts
    // also count the number of active threads in order to process the entire data range
    for (int i = 0; i < dataSize; i++)
    {
        int pixel = 0;
        // ts32fC4_Aligned
        if (ts32fC4_Aligned.ThreadIsInRange(i))
        {
            pixel = ts32fC4_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC4Aligned);
            if (ts32fC4_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC4Aligned += 1; // tupel size
                threadsC4AlignedAligned++;
            }
            else
            {
                pixelsC4Aligned += 1; // one single pixel per thread
                threadsC4AlignedUnaligned++;
            }
        }

        // ts32fC4_Unaligned
        if (ts32fC4_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts32fC4_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC4Unaligned);
            if (ts32fC4_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC4Unaligned += 1; // tupel size
                threadsC4UnalignedAligned++;
            }
            else
            {
                pixelsC4Unaligned += 1; // one single pixel per thread
                threadsC4UnalignedUnaligned++;
            }
        }

        // ts32fC3_Aligned
        if (ts32fC3_Aligned.ThreadIsInRange(i))
        {
            pixel = ts32fC3_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC3Aligned);
            if (ts32fC3_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC3Aligned += 1; // tupel size
                threadsC3AlignedAligned++;
            }
            else
            {
                pixelsC3Aligned += 1; // one single pixel per thread
                threadsC3AlignedUnaligned++;
            }
        }

        // ts32fC3_Unaligned
        if (ts32fC3_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts32fC3_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC3Unaligned);
            if (ts32fC3_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC3Unaligned += 1; // tupel size
                threadsC3UnalignedAligned++;
            }
            else
            {
                pixelsC3Unaligned += 1; // one single pixel per thread
                threadsC3UnalignedUnaligned++;
            }
        }

        // ts32fC2_Aligned
        if (ts32fC2_Aligned.ThreadIsInRange(i))
        {
            pixel = ts32fC2_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC2Aligned);
            if (ts32fC2_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC2Aligned += 1; // tupel size
                threadsC2AlignedAligned++;
            }
            else
            {
                pixelsC2Aligned += 1; // one single pixel per thread
                threadsC2AlignedUnaligned++;
            }
        }

        // ts32fC2_Unaligned
        if (ts32fC2_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts32fC2_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC2Unaligned);
            if (ts32fC2_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC2Unaligned += 1; // tupel size
                threadsC2UnalignedAligned++;
            }
            else
            {
                pixelsC2Unaligned += 1; // one single pixel per thread
                threadsC2UnalignedUnaligned++;
            }
        }

        // ts32fC1_Aligned
        if (ts32fC1_Aligned.ThreadIsInRange(i))
        {
            pixel = ts32fC1_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC1Aligned);
            if (ts32fC1_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC1Aligned += 2; // tupel size
                threadsC1AlignedAligned++;
            }
            else
            {
                pixelsC1Aligned += 1; // one single pixel per thread
                threadsC1AlignedUnaligned++;
            }
        }

        // ts32fC1_Unaligned
        if (ts32fC1_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts32fC1_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC1Unaligned);
            if (ts32fC1_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC1Unaligned += 2; // tupel size
                threadsC1UnalignedAligned++;
            }
            else
            {
                pixelsC1Unaligned += 1; // one single pixel per thread
                threadsC1UnalignedUnaligned++;
            }
        }
    }

    CHECK(pixelsC4Aligned == dataSize);
    CHECK(pixelsC4Unaligned == dataSize);

    CHECK(pixelsC3Aligned == dataSize);
    CHECK(pixelsC3Unaligned == dataSize);

    CHECK(pixelsC2Aligned == dataSize);
    CHECK(pixelsC2Unaligned == dataSize);

    CHECK(pixelsC1Aligned == dataSize);
    CHECK(pixelsC1Unaligned == dataSize);

    CHECK(threadsC4AlignedAligned == dataSize);
    CHECK(threadsC4AlignedUnaligned == 0);
    CHECK(threadsC4UnalignedAligned == dataSize);
    CHECK(threadsC4UnalignedUnaligned == 0);

    CHECK(threadsC3AlignedAligned == dataSize);
    CHECK(threadsC3AlignedUnaligned == 0);
    CHECK(threadsC3UnalignedAligned == dataSize);
    CHECK(threadsC3UnalignedUnaligned == 0);

    CHECK(threadsC2AlignedAligned == dataSize);
    CHECK(threadsC2AlignedUnaligned == 0);
    CHECK(threadsC2UnalignedAligned == dataSize);
    CHECK(threadsC2UnalignedUnaligned == 0);

    CHECK(threadsC1AlignedAligned == dataSize / 2);
    CHECK(threadsC1AlignedUnaligned == 0);
    CHECK(threadsC1UnalignedAligned == (dataSize - bytesPerWarp / int(sizeof(Pixel32fC1))) / 2);
    CHECK(threadsC1UnalignedUnaligned == bytesPerWarp / int(sizeof(Pixel32fC1)));
}

TEST_CASE("ThreadSplit - 2 bytes base type", "[Common.Image]")
{
    Pixel16sC4 *vec16sC4 = reinterpret_cast<Pixel16sC4 *>(address);
    Pixel16sC3 *vec16sC3 = reinterpret_cast<Pixel16sC3 *>(address);
    Pixel16sC2 *vec16sC2 = reinterpret_cast<Pixel16sC2 *>(address);
    Pixel16sC1 *vec16sC1 = reinterpret_cast<Pixel16sC1 *>(address);

    // Vector3 has 6 bytes and can't be a multiple of bytesPerWarp, impossible to align
    CHECK_THROWS_AS((ThreadSplit<bytesPerWarp, 2>(vec16sC3, dataSize)), InvalidArgumentException);

    CHECK_NOTHROW(ThreadSplit<bytesPerWarp, 2>(vec16sC4, dataSize));
    CHECK_NOTHROW(ThreadSplit<bytesPerWarp, 2>(vec16sC1, dataSize));

    ThreadSplit<bytesPerWarp, 1> ts16sC4_Aligned(vec16sC4, dataSize);
    ThreadSplit<bytesPerWarp, 1> ts16sC4_Unaligned(vec16sC4 + 1, dataSize);

    ThreadSplit<bytesPerWarp, 1> ts16sC3_Aligned(vec16sC3, dataSize);
    ThreadSplit<bytesPerWarp, 1> ts16sC3_Unaligned(vec16sC3 + 1, dataSize);

    ThreadSplit<bytesPerWarp, 2> ts16sC2_Aligned(vec16sC2, dataSize);
    ThreadSplit<bytesPerWarp, 2> ts16sC2_Unaligned(vec16sC2 + 1, dataSize);

    ThreadSplit<bytesPerWarp, 4> ts16sC1_Aligned(vec16sC1, dataSize);
    ThreadSplit<bytesPerWarp, 4> ts16sC1_Unaligned(vec16sC1 + 1, dataSize);

    CHECK(ts16sC4_Aligned.Muted() == 0);
    CHECK(ts16sC4_Aligned.Left() == 0);
    CHECK(ts16sC4_Aligned.Center() == dataSize);
    CHECK(ts16sC4_Aligned.Right() == 0);

    CHECK(ts16sC4_Unaligned.Muted() == 0);
    CHECK(ts16sC4_Unaligned.Left() == 0);
    CHECK(ts16sC4_Unaligned.Center() == dataSize);
    CHECK(ts16sC4_Unaligned.Right() == 0);

    CHECK(ts16sC3_Aligned.Muted() == 0);
    CHECK(ts16sC3_Aligned.Left() == 0);
    CHECK(ts16sC3_Aligned.Center() == dataSize);
    CHECK(ts16sC3_Aligned.Right() == 0);

    CHECK(ts16sC3_Unaligned.Muted() == 0);
    CHECK(ts16sC3_Unaligned.Left() == 0);
    CHECK(ts16sC3_Unaligned.Center() == dataSize);
    CHECK(ts16sC3_Unaligned.Right() == 0);

    CHECK(ts16sC2_Aligned.Muted() == 0);
    CHECK(ts16sC2_Aligned.Left() == 0);
    CHECK(ts16sC2_Aligned.Center() == dataSize / 2);
    CHECK(ts16sC2_Aligned.Right() == 0);

    CHECK(ts16sC2_Unaligned.Muted() == 1);
    CHECK(ts16sC2_Unaligned.Left() == 63);
    CHECK(ts16sC2_Unaligned.Center() == (dataSize - (bytesPerWarp / to_int(sizeof(Pixel16sC2)))) / 2);
    CHECK(ts16sC2_Unaligned.Right() == 1);
    CHECK(ts16sC2_Unaligned.Left() * to_int(sizeof(Pixel16sC2)) < bytesPerWarp);
    CHECK(ts16sC2_Unaligned.Right() * to_int(sizeof(Pixel16sC2)) < bytesPerWarp);

    CHECK(ts16sC1_Aligned.Muted() == 0);
    CHECK(ts16sC1_Aligned.Left() == 0);
    CHECK(ts16sC1_Aligned.Center() == dataSize / 4);
    CHECK(ts16sC1_Aligned.Right() == 0);

    CHECK(ts16sC1_Unaligned.Muted() == 1);
    CHECK(ts16sC1_Unaligned.Left() == 127);
    CHECK(ts16sC1_Unaligned.Center() == (dataSize - (bytesPerWarp / to_int(sizeof(Pixel16sC1)))) / 4);
    CHECK(ts16sC1_Unaligned.Right() == 1);
    CHECK(ts16sC1_Unaligned.Left() * to_int(sizeof(Pixel16sC1)) < bytesPerWarp);
    CHECK(ts16sC1_Unaligned.Right() * to_int(sizeof(Pixel16sC1)) < bytesPerWarp);

    int pixelsC4Aligned             = 0;
    int threadsC4AlignedAligned     = 0;
    int threadsC4AlignedUnaligned   = 0;
    int pixelsC4Unaligned           = 0;
    int threadsC4UnalignedAligned   = 0;
    int threadsC4UnalignedUnaligned = 0;

    int pixelsC3Aligned             = 0;
    int threadsC3AlignedAligned     = 0;
    int threadsC3AlignedUnaligned   = 0;
    int pixelsC3Unaligned           = 0;
    int threadsC3UnalignedAligned   = 0;
    int threadsC3UnalignedUnaligned = 0;

    int pixelsC2Aligned             = 0;
    int threadsC2AlignedAligned     = 0;
    int threadsC2AlignedUnaligned   = 0;
    int pixelsC2Unaligned           = 0;
    int threadsC2UnalignedAligned   = 0;
    int threadsC2UnalignedUnaligned = 0;

    int pixelsC1Aligned             = 0;
    int threadsC1AlignedAligned     = 0;
    int threadsC1AlignedUnaligned   = 0;
    int pixelsC1Unaligned           = 0;
    int threadsC1UnalignedAligned   = 0;
    int threadsC1UnalignedUnaligned = 0;

    // i in this loop represents a processing thread (CUDA thread)
    // count the total number of processed pixels in aligned and unaligned splitting parts
    // also count the number of active threads in order to process the entire data range
    for (int i = 0; i < dataSize; i++)
    {
        int pixel = 0;
        // ts16sC4_Aligned
        if (ts16sC4_Aligned.ThreadIsInRange(i))
        {
            pixel = ts16sC4_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC4Aligned);
            if (ts16sC4_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC4Aligned += 1; // tupel size
                threadsC4AlignedAligned++;
            }
            else
            {
                pixelsC4Aligned += 1; // one single pixel per thread
                threadsC4AlignedUnaligned++;
            }
        }

        // ts16sC4_Unaligned
        if (ts16sC4_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts16sC4_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC4Unaligned);
            if (ts16sC4_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC4Unaligned += 1; // tupel size
                threadsC4UnalignedAligned++;
            }
            else
            {
                pixelsC4Unaligned += 1; // one single pixel per thread
                threadsC4UnalignedUnaligned++;
            }
        }

        // ts16sC3_Aligned
        if (ts16sC3_Aligned.ThreadIsInRange(i))
        {
            pixel = ts16sC3_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC3Aligned);
            if (ts16sC3_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC3Aligned += 1; // tupel size
                threadsC3AlignedAligned++;
            }
            else
            {
                pixelsC3Aligned += 1; // one single pixel per thread
                threadsC3AlignedUnaligned++;
            }
        }

        // ts16sC3_Unaligned
        if (ts16sC3_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts16sC3_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC3Unaligned);
            if (ts16sC3_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC3Unaligned += 1; // tupel size
                threadsC3UnalignedAligned++;
            }
            else
            {
                pixelsC3Unaligned += 1; // one single pixel per thread
                threadsC3UnalignedUnaligned++;
            }
        }

        // ts16sC2_Aligned
        if (ts16sC2_Aligned.ThreadIsInRange(i))
        {
            pixel = ts16sC2_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC2Aligned);
            if (ts16sC2_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC2Aligned += 2; // tupel size
                threadsC2AlignedAligned++;
            }
            else
            {
                pixelsC2Aligned += 1; // one single pixel per thread
                threadsC2AlignedUnaligned++;
            }
        }

        // ts16sC2_Unaligned
        if (ts16sC2_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts16sC2_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC2Unaligned);
            if (ts16sC2_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC2Unaligned += 2; // tupel size
                threadsC2UnalignedAligned++;
            }
            else
            {
                pixelsC2Unaligned += 1; // one single pixel per thread
                threadsC2UnalignedUnaligned++;
            }
        }

        // ts16sC1_Aligned
        if (ts16sC1_Aligned.ThreadIsInRange(i))
        {
            pixel = ts16sC1_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC1Aligned);
            if (ts16sC1_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC1Aligned += 4; // tupel size
                threadsC1AlignedAligned++;
            }
            else
            {
                pixelsC1Aligned += 1; // one single pixel per thread
                threadsC1AlignedUnaligned++;
            }
        }

        // ts16sC1_Unaligned
        if (ts16sC1_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts16sC1_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC1Unaligned);
            if (ts16sC1_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC1Unaligned += 4; // tupel size
                threadsC1UnalignedAligned++;
            }
            else
            {
                pixelsC1Unaligned += 1; // one single pixel per thread
                threadsC1UnalignedUnaligned++;
            }
        }
    }

    CHECK(pixelsC4Aligned == dataSize);
    CHECK(pixelsC4Unaligned == dataSize);

    CHECK(pixelsC3Aligned == dataSize);
    CHECK(pixelsC3Unaligned == dataSize);

    CHECK(pixelsC2Aligned == dataSize);
    CHECK(pixelsC2Unaligned == dataSize);

    CHECK(pixelsC1Aligned == dataSize);
    CHECK(pixelsC1Unaligned == dataSize);

    CHECK(threadsC4AlignedAligned == dataSize);
    CHECK(threadsC4AlignedUnaligned == 0);
    CHECK(threadsC4UnalignedAligned == dataSize);
    CHECK(threadsC4UnalignedUnaligned == 0);

    CHECK(threadsC3AlignedAligned == dataSize);
    CHECK(threadsC3AlignedUnaligned == 0);
    CHECK(threadsC3UnalignedAligned == dataSize);
    CHECK(threadsC3UnalignedUnaligned == 0);

    CHECK(threadsC2AlignedAligned == dataSize / 2);
    CHECK(threadsC2AlignedUnaligned == 0);
    CHECK(threadsC2UnalignedAligned == (dataSize - bytesPerWarp / int(sizeof(Pixel16sC2))) / 2);
    CHECK(threadsC2UnalignedUnaligned == bytesPerWarp / int(sizeof(Pixel16sC2)));

    CHECK(threadsC1AlignedAligned == dataSize / 4);
    CHECK(threadsC1AlignedUnaligned == 0);
    CHECK(threadsC1UnalignedAligned == (dataSize - bytesPerWarp / int(sizeof(Pixel16sC1))) / 4);
    CHECK(threadsC1UnalignedUnaligned == bytesPerWarp / int(sizeof(Pixel16sC1)));
}

TEST_CASE("ThreadSplit - 1 byte base type", "[Common.Image]")
{
    Pixel8uC4 *vec8uC4 = reinterpret_cast<Pixel8uC4 *>(address);
    Pixel8uC3 *vec8uC3 = reinterpret_cast<Pixel8uC3 *>(address);
    Pixel8uC2 *vec8uC2 = reinterpret_cast<Pixel8uC2 *>(address);
    Pixel8uC1 *vec8uC1 = reinterpret_cast<Pixel8uC1 *>(address);

    // Vector3 has 3 bytes and can't be a multiple of bytesPerWarp, impossible to align
    CHECK_THROWS_AS((ThreadSplit<bytesPerWarp, 2>(vec8uC3, dataSize)), InvalidArgumentException);

    CHECK_NOTHROW(ThreadSplit<bytesPerWarp, 2>(vec8uC4, dataSize));
    CHECK_NOTHROW(ThreadSplit<bytesPerWarp, 4>(vec8uC1, dataSize));

    ThreadSplit<bytesPerWarp, 2> ts8uC4_Aligned(vec8uC4, dataSize);
    ThreadSplit<bytesPerWarp, 2> ts8uC4_Unaligned(vec8uC4 + 1, dataSize);

    ThreadSplit<bytesPerWarp, 1> ts8uC3_Aligned(vec8uC3, dataSize);
    ThreadSplit<bytesPerWarp, 1> ts8uC3_Unaligned(vec8uC3 + 1, dataSize);

    ThreadSplit<bytesPerWarp, 4> ts8uC2_Aligned(vec8uC2, dataSize);
    ThreadSplit<bytesPerWarp, 4> ts8uC2_Unaligned(vec8uC2 + 1, dataSize);

    ThreadSplit<bytesPerWarp, 8> ts8uC1_Aligned(vec8uC1, dataSize);
    ThreadSplit<bytesPerWarp, 8> ts8uC1_Unaligned(vec8uC1 + 1, dataSize);

    CHECK(ts8uC4_Aligned.Muted() == 0);
    CHECK(ts8uC4_Aligned.Left() == 0);
    CHECK(ts8uC4_Aligned.Center() == dataSize / 2);
    CHECK(ts8uC4_Aligned.Right() == 0);

    CHECK(ts8uC4_Unaligned.Muted() == 1);
    CHECK(ts8uC4_Unaligned.Left() == 63);
    CHECK(ts8uC4_Unaligned.Center() == (dataSize - (bytesPerWarp / to_int(sizeof(Pixel8uC4)))) / 2);
    CHECK(ts8uC4_Unaligned.Right() == 1);
    CHECK(ts8uC4_Unaligned.Left() * to_int(sizeof(Pixel8uC4)) < bytesPerWarp);
    CHECK(ts8uC4_Unaligned.Right() * to_int(sizeof(Pixel8uC4)) < bytesPerWarp);

    CHECK(ts8uC3_Aligned.Muted() == 0);
    CHECK(ts8uC3_Aligned.Left() == 0);
    CHECK(ts8uC3_Aligned.Center() == dataSize);
    CHECK(ts8uC3_Aligned.Right() == 0);

    CHECK(ts8uC3_Unaligned.Muted() == 0);
    CHECK(ts8uC3_Unaligned.Left() == 0);
    CHECK(ts8uC3_Unaligned.Center() == dataSize);
    CHECK(ts8uC3_Unaligned.Right() == 0);

    CHECK(ts8uC2_Aligned.Muted() == 0);
    CHECK(ts8uC2_Aligned.Left() == 0);
    CHECK(ts8uC2_Aligned.Center() == dataSize / 4);
    CHECK(ts8uC2_Aligned.Right() == 0);

    CHECK(ts8uC2_Unaligned.Muted() == 1);
    CHECK(ts8uC2_Unaligned.Left() == 127);
    CHECK(ts8uC2_Unaligned.Center() == (dataSize - (bytesPerWarp / to_int(sizeof(Pixel8uC2)))) / 4);
    CHECK(ts8uC2_Unaligned.Right() == 1);
    CHECK(ts8uC4_Unaligned.Left() * to_int(sizeof(Pixel8uC2)) < bytesPerWarp);
    CHECK(ts8uC4_Unaligned.Right() * to_int(sizeof(Pixel8uC2)) < bytesPerWarp);

    CHECK(ts8uC1_Aligned.Muted() == 0);
    CHECK(ts8uC1_Aligned.Left() == 0);
    CHECK(ts8uC1_Aligned.Center() == dataSize / 8);
    CHECK(ts8uC1_Aligned.Right() == 0);

    CHECK(ts8uC1_Unaligned.Muted() == 1);
    CHECK(ts8uC1_Unaligned.Left() == 255);
    CHECK(ts8uC1_Unaligned.Center() == (dataSize - (bytesPerWarp / to_int(sizeof(Pixel8uC1)))) / 8);
    CHECK(ts8uC1_Unaligned.Right() == 1);
    CHECK(ts8uC4_Unaligned.Left() * to_int(sizeof(Pixel8uC1)) < bytesPerWarp);
    CHECK(ts8uC4_Unaligned.Right() * to_int(sizeof(Pixel8uC1)) < bytesPerWarp);

    int pixelsC4Aligned             = 0;
    int threadsC4AlignedAligned     = 0;
    int threadsC4AlignedUnaligned   = 0;
    int pixelsC4Unaligned           = 0;
    int threadsC4UnalignedAligned   = 0;
    int threadsC4UnalignedUnaligned = 0;

    int pixelsC3Aligned             = 0;
    int threadsC3AlignedAligned     = 0;
    int threadsC3AlignedUnaligned   = 0;
    int pixelsC3Unaligned           = 0;
    int threadsC3UnalignedAligned   = 0;
    int threadsC3UnalignedUnaligned = 0;

    int pixelsC2Aligned             = 0;
    int threadsC2AlignedAligned     = 0;
    int threadsC2AlignedUnaligned   = 0;
    int pixelsC2Unaligned           = 0;
    int threadsC2UnalignedAligned   = 0;
    int threadsC2UnalignedUnaligned = 0;

    int pixelsC1Aligned             = 0;
    int threadsC1AlignedAligned     = 0;
    int threadsC1AlignedUnaligned   = 0;
    int pixelsC1Unaligned           = 0;
    int threadsC1UnalignedAligned   = 0;
    int threadsC1UnalignedUnaligned = 0;

    // i in this loop represents a processing thread (CUDA thread)
    // count the total number of processed pixels in aligned and unaligned splitting parts
    // also count the number of active threads in order to process the entire data range
    for (int i = 0; i < dataSize; i++)
    {
        int pixel = 0;
        // ts8uC4_Aligned
        if (ts8uC4_Aligned.ThreadIsInRange(i))
        {
            pixel = ts8uC4_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC4Aligned);
            if (ts8uC4_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC4Aligned += 2; // tupel size
                threadsC4AlignedAligned++;
            }
            else
            {
                pixelsC4Aligned += 1; // one single pixel per thread
                threadsC4AlignedUnaligned++;
            }
        }

        // ts8uC4_Unaligned
        if (ts8uC4_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts8uC4_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC4Unaligned);
            if (ts8uC4_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC4Unaligned += 2; // tupel size
                threadsC4UnalignedAligned++;
            }
            else
            {
                pixelsC4Unaligned += 1; // one single pixel per thread
                threadsC4UnalignedUnaligned++;
            }
        }

        // ts8uC3_Aligned
        if (ts8uC3_Aligned.ThreadIsInRange(i))
        {
            pixel = ts8uC3_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC3Aligned);
            if (ts8uC3_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC3Aligned += 1; // tupel size
                threadsC3AlignedAligned++;
            }
            else
            {
                pixelsC3Aligned += 1; // one single pixel per thread
                threadsC3AlignedUnaligned++;
            }
        }

        // ts8uC3_Unaligned
        if (ts8uC3_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts8uC3_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC3Unaligned);
            if (ts8uC3_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC3Unaligned += 1; // tupel size
                threadsC3UnalignedAligned++;
            }
            else
            {
                pixelsC3Unaligned += 1; // one single pixel per thread
                threadsC3UnalignedUnaligned++;
            }
        }

        // ts8uC2_Aligned
        if (ts8uC2_Aligned.ThreadIsInRange(i))
        {
            pixel = ts8uC2_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC2Aligned);
            if (ts8uC2_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC2Aligned += 4; // tupel size
                threadsC2AlignedAligned++;
            }
            else
            {
                pixelsC2Aligned += 1; // one single pixel per thread
                threadsC2AlignedUnaligned++;
            }
        }

        // ts8uC2_Unaligned
        if (ts8uC2_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts8uC2_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC2Unaligned);
            if (ts8uC2_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC2Unaligned += 4; // tupel size
                threadsC2UnalignedAligned++;
            }
            else
            {
                pixelsC2Unaligned += 1; // one single pixel per thread
                threadsC2UnalignedUnaligned++;
            }
        }

        // ts8uC1_Aligned
        if (ts8uC1_Aligned.ThreadIsInRange(i))
        {
            pixel = ts8uC1_Aligned.GetPixel(i);
            CHECK(pixel == pixelsC1Aligned);
            if (ts8uC1_Aligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC1Aligned += 8; // tupel size
                threadsC1AlignedAligned++;
            }
            else
            {
                pixelsC1Aligned += 1; // one single pixel per thread
                threadsC1AlignedUnaligned++;
            }
        }

        // ts8uC1_Unaligned
        if (ts8uC1_Unaligned.ThreadIsInRange(i))
        {
            pixel = ts8uC1_Unaligned.GetPixel(i);
            CHECK(pixel == pixelsC1Unaligned);
            if (ts8uC1_Unaligned.ThreadIsAlignedToWarp(i))
            {
                pixelsC1Unaligned += 8; // tupel size
                threadsC1UnalignedAligned++;
            }
            else
            {
                pixelsC1Unaligned += 1; // one single pixel per thread
                threadsC1UnalignedUnaligned++;
            }
        }
    }

    CHECK(pixelsC4Aligned == dataSize);
    CHECK(pixelsC4Unaligned == dataSize);

    CHECK(pixelsC3Aligned == dataSize);
    CHECK(pixelsC3Unaligned == dataSize);

    CHECK(pixelsC2Aligned == dataSize);
    CHECK(pixelsC2Unaligned == dataSize);

    CHECK(pixelsC1Aligned == dataSize);
    CHECK(pixelsC1Unaligned == dataSize);

    CHECK(threadsC4AlignedAligned == dataSize / 2);
    CHECK(threadsC4AlignedUnaligned == 0);
    CHECK(threadsC4UnalignedAligned == (dataSize - bytesPerWarp / int(sizeof(Pixel8uC4))) / 2);
    CHECK(threadsC4UnalignedUnaligned == bytesPerWarp / int(sizeof(Pixel8uC4)));

    CHECK(threadsC3AlignedAligned == dataSize);
    CHECK(threadsC3AlignedUnaligned == 0);
    CHECK(threadsC3UnalignedAligned == dataSize);
    CHECK(threadsC3UnalignedUnaligned == 0);

    CHECK(threadsC2AlignedAligned == dataSize / 4);
    CHECK(threadsC2AlignedUnaligned == 0);
    CHECK(threadsC2UnalignedAligned == (dataSize - bytesPerWarp / int(sizeof(Pixel8uC2))) / 4);
    CHECK(threadsC2UnalignedUnaligned == bytesPerWarp / int(sizeof(Pixel8uC2)));

    CHECK(threadsC1AlignedAligned == dataSize / 8);
    CHECK(threadsC1AlignedUnaligned == 0);
    CHECK(threadsC1UnalignedAligned == (dataSize - bytesPerWarp / int(sizeof(Pixel8uC1))) / 8);
    CHECK(threadsC1UnalignedUnaligned == bytesPerWarp / int(sizeof(Pixel8uC1)));
}
