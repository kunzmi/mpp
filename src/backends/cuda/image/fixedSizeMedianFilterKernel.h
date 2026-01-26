#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/gotoPtr.h>
#include <common/image/pitchException.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vector2.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{

// The implementation is mainly inspired by the two following papers:
// Fast median filters using separable sorting networks (https://dl.acm.org/doi/epdf/10.1145/3450626.3459773)
// and to some extent
// Efficient GPU-based implementation of the median filter based on a multi-pixel-per-thread framework by Salvador et al
// There are more recent and more performant publications / implementations available, but for the time beeing we should
// keep it simple. And being more than 100x faster than NPP is not that bad...

template <typename T> __device__ void sort2(T &aA, T &aB)
{
    T min = T::Min(aA, aB);
    aB    = T::Max(aA, aB);
    aA    = min;
}

template <typename T, int filterSize> class Median
{
};

template <typename T> class Median<T, 3>
{
  public:
    __device__ static void sort(T aData[3])
    {
        // see: https://github.com/bertdobbelaere/SorterHunter/blob/master/Networks/Sorters/Sort_3_3_3.json
        sort2(aData[0], aData[2]);
        sort2(aData[0], aData[1]);
        sort2(aData[1], aData[2]);
    }

    template <int extendedBlockW> __device__ static T median(const T aColSorted[3][extendedBlockW], int aXOffset)
    {
        T center[3] = {aColSorted[1][0 + aXOffset], aColSorted[1][1 + aXOffset], aColSorted[1][2 + aXOffset]};
        sort(center);

        center[0] =
            T::Max(T::Max(aColSorted[0][0 + aXOffset], aColSorted[0][1 + aXOffset]), aColSorted[0][2 + aXOffset]);
        center[2] =
            T::Min(T::Min(aColSorted[2][0 + aXOffset], aColSorted[2][1 + aXOffset]), aColSorted[2][2 + aXOffset]);
        sort(center);

        return center[1];
    }
};

template <typename T> class Median<T, 5>
{
  public:
    __device__ static void sort(T aData[5])
    {
        // see: https://github.com/bertdobbelaere/SorterHunter/blob/master/Networks/Sorters/Sort_5_9_5.json
        sort2(aData[0], aData[3]);
        sort2(aData[1], aData[4]);
        sort2(aData[0], aData[2]);
        sort2(aData[1], aData[3]);
        sort2(aData[0], aData[1]);
        sort2(aData[2], aData[4]);
        sort2(aData[1], aData[2]);
        sort2(aData[3], aData[4]);
        sort2(aData[2], aData[3]);
    }

    __device__ static void max2(T aData[5])
    {
        // see:
        // https://github.com/bertdobbelaere/SorterHunter/blob/master/Networks/Sorters_Low_Avg_Swaps/Sort_LS_5_9_5.json
        // swap 0-1 and swap 1-2 is pruned
        sort2(aData[0], aData[4]);
        sort2(aData[0], aData[2]);
        sort2(aData[1], aData[4]);
        sort2(aData[1], aData[3]);
        sort2(aData[2], aData[4]);
        sort2(aData[2], aData[3]);
        sort2(aData[3], aData[4]);
    }

    __device__ static void min2(T aData[5])
    {
        // see:
        // https://github.com/bertdobbelaere/SorterHunter/blob/master/Networks/Sorters_Low_Avg_Swaps/Sort_LS_5_9_5.json
        // swap 0-1 and swap 1-2 is pruned
        // mirrored to max
        sort2(aData[0], aData[4]);
        sort2(aData[2], aData[4]);
        sort2(aData[0], aData[3]);
        sort2(aData[1], aData[3]);
        sort2(aData[0], aData[2]);
        sort2(aData[1], aData[2]);
        sort2(aData[0], aData[1]);
    }

    __device__ static void center1(T aData[5])
    {
        // see:
        // https://github.com/bertdobbelaere/SorterHunter/blob/master/Networks/Median/Median_5_7_5.json
        sort2(aData[0], aData[1]);
        sort2(aData[2], aData[3]);
        sort2(aData[0], aData[2]);
        sort2(aData[1], aData[3]);
        sort2(aData[2], aData[4]);
        sort2(aData[1], aData[2]);
        sort2(aData[2], aData[4]);
    }

    template <int extendedBlockW> __device__ static T median(const T aColSorted[5][extendedBlockW], int aXOffset)
    {
        T p[3];
        T diagonal[5];
        T line[5];
        // first line, we only need the two max values:
        line[0] = aColSorted[0][0 + aXOffset];
        line[1] = aColSorted[0][1 + aXOffset];
        line[2] = aColSorted[0][2 + aXOffset];
        line[3] = aColSorted[0][3 + aXOffset];
        line[4] = aColSorted[0][4 + aXOffset];
        max2(line);
        diagonal[4] = line[4];
        p[0]        = line[3];

        // second line, we need three max values, do full sort:
        line[0] = aColSorted[1][0 + aXOffset];
        line[1] = aColSorted[1][1 + aXOffset];
        line[2] = aColSorted[1][2 + aXOffset];
        line[3] = aColSorted[1][3 + aXOffset];
        line[4] = aColSorted[1][4 + aXOffset];
        sort(line);
        diagonal[3] = line[3];
        p[2]        = line[4];
        p[0]        = T::Max(p[0], line[2]);

        // center line, we need central three values, do full sort:
        line[0] = aColSorted[2][0 + aXOffset];
        line[1] = aColSorted[2][1 + aXOffset];
        line[2] = aColSorted[2][2 + aXOffset];
        line[3] = aColSorted[2][3 + aXOffset];
        line[4] = aColSorted[2][4 + aXOffset];
        sort(line);
        diagonal[2] = line[2];
        p[2]        = T::Min(p[2], line[3]);
        p[0]        = T::Max(p[0], line[1]);

        // 4th line, we need three min values, do full sort:
        line[0] = aColSorted[3][0 + aXOffset];
        line[1] = aColSorted[3][1 + aXOffset];
        line[2] = aColSorted[3][2 + aXOffset];
        line[3] = aColSorted[3][3 + aXOffset];
        line[4] = aColSorted[3][4 + aXOffset];
        sort(line);
        diagonal[1] = line[1];
        p[2]        = T::Min(p[2], line[2]);
        p[0]        = T::Max(p[0], line[0]);

        // last line, we need the two min values:
        line[0] = aColSorted[4][0 + aXOffset];
        line[1] = aColSorted[4][1 + aXOffset];
        line[2] = aColSorted[4][2 + aXOffset];
        line[3] = aColSorted[4][3 + aXOffset];
        line[4] = aColSorted[4][4 + aXOffset];
        min2(line);
        diagonal[0] = line[0];
        p[2]        = T::Min(p[2], line[1]);

        // median of diagonal:
        center1(diagonal);
        p[1] = diagonal[2];

        Median<T, 3>::sort(p);

        return p[1];
    }
};

template <typename T> class Median<T, 7>
{
  public:
    __device__ static void sort(T aData[7])
    {
        // see: https://github.com/bertdobbelaere/SorterHunter/blob/master/Networks/Sorters/Sort_7_16_6.json
        sort2(aData[0], aData[6]);
        sort2(aData[2], aData[3]);
        sort2(aData[4], aData[5]);

        sort2(aData[0], aData[2]);
        sort2(aData[1], aData[4]);
        sort2(aData[3], aData[6]);

        sort2(aData[0], aData[1]);
        sort2(aData[2], aData[5]);
        sort2(aData[3], aData[4]);

        sort2(aData[1], aData[2]);
        sort2(aData[4], aData[6]);

        sort2(aData[2], aData[3]);
        sort2(aData[4], aData[5]);

        sort2(aData[1], aData[2]);
        sort2(aData[3], aData[4]);
        sort2(aData[5], aData[6]);
    }

    __device__ static void sort6(T aData[6])
    {
        // see: https://github.com/bertdobbelaere/SorterHunter/blob/master/Networks/Sorters/Sort_6_12_5.json
        sort2(aData[0], aData[5]);
        sort2(aData[1], aData[3]);
        sort2(aData[2], aData[4]);

        sort2(aData[1], aData[2]);
        sort2(aData[3], aData[4]);

        sort2(aData[0], aData[3]);
        sort2(aData[2], aData[5]);

        sort2(aData[0], aData[1]);
        sort2(aData[2], aData[3]);
        sort2(aData[4], aData[5]);

        sort2(aData[1], aData[2]);
        sort2(aData[3], aData[4]);
    }

    __device__ static void center1(T aData[11])
    {
        // in the final step we need the median of 11 values
        // see:
        // https://github.com/bertdobbelaere/SorterHunter/blob/master/Networks/Median/Median_11_25_11.json
        sort2(aData[0], aData[1]);
        sort2(aData[2], aData[3]);
        sort2(aData[4], aData[5]);
        sort2(aData[6], aData[7]);
        sort2(aData[8], aData[9]);

        sort2(aData[0], aData[2]);
        sort2(aData[1], aData[3]);
        sort2(aData[4], aData[6]);
        sort2(aData[5], aData[7]);

        sort2(aData[1], aData[2]);
        sort2(aData[5], aData[6]);

        sort2(aData[0], aData[5]);
        sort2(aData[1], aData[4]);
        sort2(aData[2], aData[7]);
        sort2(aData[3], aData[6]);

        sort2(aData[2], aData[5]);
        sort2(aData[3], aData[4]);

        sort2(aData[2], aData[8]);
        sort2(aData[4], aData[9]);

        sort2(aData[8], aData[10]);

        sort2(aData[3], aData[8]);
        sort2(aData[5], aData[10]);

        sort2(aData[4], aData[5]);

        sort2(aData[4], aData[8]);

        sort2(aData[5], aData[8]);
    }

    __device__ static void max3(T aData[7])
    {
        // pruned Parberry’s pairwise sort network (mirrored)
        sort2(aData[5], aData[6]);
        sort2(aData[3], aData[4]);
        sort2(aData[1], aData[2]);

        sort2(aData[4], aData[6]);
        sort2(aData[0], aData[2]);

        sort2(aData[0], aData[4]);

        sort2(aData[2], aData[6]);
        sort2(aData[2], aData[4]);
        sort2(aData[3], aData[5]);

        sort2(aData[1], aData[5]);
        sort2(aData[1], aData[3]);

        sort2(aData[2], aData[5]);
        sort2(aData[4], aData[5]);
    }

    __device__ static void min3(T aData[7])
    {
        // pruned Parberry’s pairwise sort network
        sort2(aData[0], aData[1]);
        sort2(aData[2], aData[3]);
        sort2(aData[4], aData[5]);

        sort2(aData[0], aData[2]);
        sort2(aData[4], aData[6]);

        sort2(aData[2], aData[6]);

        sort2(aData[0], aData[4]);
        sort2(aData[2], aData[4]);
        sort2(aData[1], aData[3]);

        sort2(aData[1], aData[5]);
        sort2(aData[3], aData[5]);

        sort2(aData[1], aData[4]);
        sort2(aData[1], aData[2]);
    }

    __device__ static void max4(T aData[7])
    {
        // pruned Parberry’s pairwise sort network
        sort2(aData[0], aData[1]);
        sort2(aData[2], aData[3]);
        sort2(aData[4], aData[5]);

        sort2(aData[0], aData[2]);
        sort2(aData[4], aData[6]);

        sort2(aData[2], aData[6]);

        sort2(aData[0], aData[4]);
        sort2(aData[2], aData[4]);
        sort2(aData[1], aData[3]);

        sort2(aData[1], aData[5]);
        sort2(aData[3], aData[5]);

        sort2(aData[1], aData[4]);

        sort2(aData[3], aData[6]);

        // sort2(aData[1], aData[2]);
        sort2(aData[3], aData[4]);
        sort2(aData[5], aData[6]);
    }

    __device__ static void min4(T aData[7])
    {
        // pruned Parberry’s pairwise sort network
        sort2(aData[0], aData[1]);
        sort2(aData[2], aData[3]);
        sort2(aData[4], aData[5]);

        sort2(aData[0], aData[2]);
        sort2(aData[4], aData[6]);

        sort2(aData[2], aData[6]);

        sort2(aData[0], aData[4]);
        sort2(aData[2], aData[4]);
        sort2(aData[1], aData[3]);

        sort2(aData[1], aData[5]);
        sort2(aData[3], aData[5]);

        sort2(aData[1], aData[4]);

        sort2(aData[3], aData[6]);

        sort2(aData[1], aData[2]);
        sort2(aData[3], aData[4]);
        // sort2(aData[5], aData[6]);
    }

    template <int extendedBlockW> __device__ static T median(const T aColSorted[7][extendedBlockW], int aXOffset)
    {
        T remain[11];
        T diagonal0[6];
        T diagonal[7];
        T diagonal1[6];
        T line[7];

        // first line, we only need the three max values:
        line[0] = aColSorted[0][0 + aXOffset];
        line[1] = aColSorted[0][1 + aXOffset];
        line[2] = aColSorted[0][2 + aXOffset];
        line[3] = aColSorted[0][3 + aXOffset];
        line[4] = aColSorted[0][4 + aXOffset];
        line[5] = aColSorted[0][5 + aXOffset];
        line[6] = aColSorted[0][6 + aXOffset];
        max3(line);
        diagonal0[5] = line[5];
        diagonal[6]  = line[6];
        remain[9]    = line[4];

        // second line, we need four max values:
        line[0] = aColSorted[1][0 + aXOffset];
        line[1] = aColSorted[1][1 + aXOffset];
        line[2] = aColSorted[1][2 + aXOffset];
        line[3] = aColSorted[1][3 + aXOffset];
        line[4] = aColSorted[1][4 + aXOffset];
        line[5] = aColSorted[1][5 + aXOffset];
        line[6] = aColSorted[1][6 + aXOffset];
        max4(line);
        diagonal0[4] = line[4];
        diagonal[5]  = line[5];
        diagonal1[5] = line[6];
        remain[9]    = T::Max(remain[9], line[3]);

        // 3rd line, we 5 max values, do full sort:
        line[0] = aColSorted[2][0 + aXOffset];
        line[1] = aColSorted[2][1 + aXOffset];
        line[2] = aColSorted[2][2 + aXOffset];
        line[3] = aColSorted[2][3 + aXOffset];
        line[4] = aColSorted[2][4 + aXOffset];
        line[5] = aColSorted[2][5 + aXOffset];
        line[6] = aColSorted[2][6 + aXOffset];
        sort(line);
        diagonal0[3] = line[3];
        diagonal[4]  = line[4];
        diagonal1[4] = line[5];
        remain[9]    = T::Max(remain[9], line[2]);
        remain[1]    = line[6];

        // 4th line, we need central 5 values, do full sort:
        line[0] = aColSorted[3][0 + aXOffset];
        line[1] = aColSorted[3][1 + aXOffset];
        line[2] = aColSorted[3][2 + aXOffset];
        line[3] = aColSorted[3][3 + aXOffset];
        line[4] = aColSorted[3][4 + aXOffset];
        line[5] = aColSorted[3][5 + aXOffset];
        line[6] = aColSorted[3][6 + aXOffset];
        sort(line);
        diagonal0[2] = line[2];
        diagonal[3]  = line[3];
        diagonal1[3] = line[4];
        remain[9]    = T::Max(remain[9], line[1]);
        remain[1]    = T::Min(remain[1], line[5]);

        // 5th line, we 5 min values, do full sort:
        line[0] = aColSorted[4][0 + aXOffset];
        line[1] = aColSorted[4][1 + aXOffset];
        line[2] = aColSorted[4][2 + aXOffset];
        line[3] = aColSorted[4][3 + aXOffset];
        line[4] = aColSorted[4][4 + aXOffset];
        line[5] = aColSorted[4][5 + aXOffset];
        line[6] = aColSorted[4][6 + aXOffset];
        sort(line);
        diagonal0[1] = line[1];
        diagonal[2]  = line[2];
        diagonal1[2] = line[3];
        remain[9]    = T::Max(remain[9], line[0]);
        remain[1]    = T::Min(remain[1], line[4]);

        // 6th line, we 4 min values:
        line[0] = aColSorted[5][0 + aXOffset];
        line[1] = aColSorted[5][1 + aXOffset];
        line[2] = aColSorted[5][2 + aXOffset];
        line[3] = aColSorted[5][3 + aXOffset];
        line[4] = aColSorted[5][4 + aXOffset];
        line[5] = aColSorted[5][5 + aXOffset];
        line[6] = aColSorted[5][6 + aXOffset];
        min4(line);
        diagonal0[0] = line[0];
        diagonal[1]  = line[1];
        diagonal1[1] = line[2];
        remain[1]    = T::Min(remain[1], line[3]);

        // last line, we 3 min values:
        line[0] = aColSorted[6][0 + aXOffset];
        line[1] = aColSorted[6][1 + aXOffset];
        line[2] = aColSorted[6][2 + aXOffset];
        line[3] = aColSorted[6][3 + aXOffset];
        line[4] = aColSorted[6][4 + aXOffset];
        line[5] = aColSorted[6][5 + aXOffset];
        line[6] = aColSorted[6][6 + aXOffset];
        min3(line);
        diagonal[0]  = line[0];
        diagonal1[0] = line[1];
        remain[1]    = T::Min(remain[1], line[2]);

        sort6(diagonal0);
        sort6(diagonal1);
        sort(diagonal);

        remain[0]  = diagonal1[0];
        remain[2]  = diagonal1[1];
        remain[4]  = diagonal1[2];
        remain[3]  = diagonal[2];
        remain[5]  = diagonal[3];
        remain[7]  = diagonal[4];
        remain[6]  = diagonal0[3];
        remain[8]  = diagonal0[4];
        remain[10] = diagonal0[5];

        center1(remain);

        return remain[5];
    }
};

/// <summary>
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int blockHeight, typename BorderControlT,
          int filterSize>
__global__ void fixedSizeMedianFilterKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                            Vector2<int> aFilterCenter, Size2D aSize,
                                            ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    constexpr int myWarpSize = 32; // warpSize itself is not const nor constexpr...

    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    // we need the full warp in x direction to do the load...
    if (threadY >= DIV_UP(aSize.y, blockHeight))
    {
        return;
    }

    extern __shared__ int sharedBuffer[];
    // ComputeT *buffer = reinterpret_cast<ComputeT *>(sharedBuffer);

    const int pixelX  = aSplit.GetPixel(threadX);
    const int pixelX0 = aSplit.GetPixel(blockIdx.x * blockDim.x);
    const int pixelY  = threadY * blockHeight;

    // don't check for warp alignment if TupelSize <= 1
    if constexpr (TupelSize > 1) // evaluated at compile time
    {
        constexpr int extendedBlockW                  = myWarpSize * TupelSize + (filterSize - 1);
        constexpr int extendedBlockH                  = blockHeight + (filterSize - 1);
        ComputeT(*buffer)[filterSize][extendedBlockW] = (ComputeT(*)[filterSize][extendedBlockW])(sharedBuffer);

        // as threads in warp-aligned area are always the full warp, no need to check for X pixel limits here
        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
#pragma unroll
            for (int i = 0; i < extendedBlockW; i += myWarpSize)
            {
                const int offsetX = i + threadIdx.x;

                if (offsetX < extendedBlockW)
                {
                    ComputeT localBuffer[extendedBlockH];

#pragma unroll
                    for (int ry = 0; ry < extendedBlockH; ry++)
                    {
                        const int srcPixelY = ry - aFilterCenter.y + pixelY;

                        const int srcPixelX = pixelX0 - aFilterCenter.x + offsetX;
                        localBuffer[ry]     = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
                    }

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        ComputeT sortBuffer[filterSize];

#pragma unroll
                        for (int sy = 0; sy < filterSize; sy++)
                        {
                            sortBuffer[sy] = localBuffer[sy + ky];
                        }

                        Median<ComputeT, filterSize>::sort(sortBuffer);

#pragma unroll
                        for (int sy = 0; sy < filterSize; sy++)
                        {
                            buffer[(ky + threadIdx.y * blockHeight)][sy][offsetX] = sortBuffer[sy];
                        }
                    }
                }
            }

            __syncthreads();

            // now we have sorted columns for the entire thread-block in shared memory

#pragma unroll
            for (int bl = 0; bl < blockHeight; bl++)
            {
                const int pixelYDst = pixelY + bl;

                if (pixelYDst < aSize.y)
                {
                    DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelYDst);
                    Tupel<DstT, TupelSize> res;

#pragma unroll
                    for (int t = 0; t < TupelSize; t++)
                    {
                        res.value[t] = Median<ComputeT, filterSize>::median<extendedBlockW>(
                            buffer[bl + threadIdx.y * blockHeight], threadIdx.x * TupelSize + t);

                        // restore alpha channel values:
                        if constexpr (has_alpha_channel_v<DstT>)
                        {
                            res.value[t].w = (pixelsOut + t)->w;
                        }
                    }
                    Tupel<DstT, TupelSize>::StoreAligned(res, pixelsOut);
                }
            }
            return;
        }
    }

    {

        constexpr int extendedBlockW                  = myWarpSize + (filterSize - 1);
        constexpr int extendedBlockH                  = blockHeight + (filterSize - 1);
        ComputeT(*buffer)[filterSize][extendedBlockW] = (ComputeT(*)[filterSize][extendedBlockW])(sharedBuffer);

#pragma unroll
        for (int i = 0; i < extendedBlockW; i += myWarpSize)
        {
            const int offsetX = i + threadIdx.x;

            if (offsetX < extendedBlockW)
            {
                ComputeT localBuffer[extendedBlockH];

#pragma unroll
                for (int ry = 0; ry < extendedBlockH; ry++)
                {
                    const int srcPixelY = ry - aFilterCenter.y + pixelY;

                    const int srcPixelX = pixelX0 - aFilterCenter.x + offsetX;
                    localBuffer[ry]     = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        ComputeT sortBuffer[filterSize];

#pragma unroll
                        for (int sy = 0; sy < filterSize; sy++)
                        {
                            sortBuffer[sy] = localBuffer[sy + ky];
                        }

                        Median<ComputeT, filterSize>::sort(sortBuffer);

#pragma unroll
                        for (int sy = 0; sy < filterSize; sy++)
                        {
                            buffer[(ky + threadIdx.y * blockHeight)][sy][offsetX] = sortBuffer[sy];
                        }
                    }
                }
            }
        }

        __syncthreads();

        // now we have sorted columns for the entire thread-block in shared memory

        // now that the entire warp has done the loading, we need to check for correct X-pixel:
        if (pixelX >= 0 && pixelX < aSize.x)
        {
#pragma unroll
            for (int bl = 0; bl < blockHeight; bl++)
            {
                const int pixelYDst = pixelY + bl;

                if (pixelYDst < aSize.y)
                {
                    DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelYDst);

                    DstT res = Median<ComputeT, filterSize>::median<extendedBlockW>(
                        buffer[bl + threadIdx.y * blockHeight], threadIdx.x);

                    // restore alpha channel values:
                    if constexpr (has_alpha_channel_v<DstT>)
                    {
                        res.w = pixelsOut->w;
                    }
                    *pixelsOut = res;
                }
            }
        }

        return;
    }
}

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, int blockHeight,
          typename BorderControlT, int filterSize>
void InvokeFixedSizeMedianFilterKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                       const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                       const Vector2<int> &aFilterCenter, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    fixedSizeMedianFilterKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, blockHeight, BorderControlT,
                                filterSize>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aFilterCenter, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int blockHeight, typename BorderControlT>
void InvokeFixedSizeMedianFilterKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                              int aFilterSize, const Vector2<int> &aFilterCenter, const Size2D &aSize,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        checkPitchIsMultiple(aPitchDst, ConfigWarpAlignment<"Default">::value, TupelSize);

        dim3 BlockSize = ConfigBlockSize<"Default">::value;

        if (blockHeight > 1)
        {
            BlockSize.y = 4;
        }

        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        const uint extendedBlockW          = BlockSize.x * TupelSize + (aFilterSize - 1);
        uint SharedMemory = sizeof(ComputeT) * extendedBlockW * blockHeight * BlockSize.y * aFilterSize;

        if (SharedMemory > aStreamCtx.SharedMemPerBlock)
        {
            BlockSize.y  = 2;
            SharedMemory = sizeof(ComputeT) * extendedBlockW * blockHeight * BlockSize.y * aFilterSize;
        }

        if (SharedMemory > aStreamCtx.SharedMemPerBlock)
        {
            throw CUDAUNSUPPORTED(fixedSizeMedianFilterKernel,
                                  "Kernel launch failed as too much shared memory was requested: "
                                      << SharedMemory << ". Available size is: " << aStreamCtx.SharedMemPerBlock
                                      << ".");
        }

        switch (aFilterSize)
        {
            case 3:
                InvokeFixedSizeMedianFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                  BorderControlT, 3>(BlockSize, SharedMemory, aStreamCtx.WarpSize,
                                                                     aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                                                                     aFilterCenter, aSize);
                break;
            case 5:
                InvokeFixedSizeMedianFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                  BorderControlT, 5>(BlockSize, SharedMemory, aStreamCtx.WarpSize,
                                                                     aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                                                                     aFilterCenter, aSize);
                break;
            case 7:
                InvokeFixedSizeMedianFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                  BorderControlT, 7>(BlockSize, SharedMemory, aStreamCtx.WarpSize,
                                                                     aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                                                                     aFilterCenter, aSize);
                break;
            default:
                throw INVALIDARGUMENT(aFilterSize,
                                      "Only sizes 3, 5, and 7 are implemented as fixed size kernels. Provided size is: "
                                          << aFilterSize << ".");
                break;
        }
    }
    else
    {
        throw CUDAUNSUPPORTED(fixedSizeMedianFilterKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
