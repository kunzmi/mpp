#pragma once
#include <common/defines.h>
#include <common/exception.h>
#include <common/safeCast.h>
#include <compare>
#include <cstddef>

namespace opp::image
{

/// <summary>
/// A ThreadSplit defines how computing threads are split over an array of data.<para/>
/// It contains a number of muted threads that are not supposed to do any computation and are only set in order to align
/// data to warp size in best possible manner.<para/> The Left part are the data elements that are not aligned to a
/// multiple of a warp.<para/> The Center part is aligned to warp size.<para/> The Right part is again not fully aligned
/// to warp size.<para/>
/// Using int as base type as split is intended to be performed on X-dimension of an image only
/// </summary>
/// <typeparam name="WarpAlignmentInBytes">How many bytes a warp processes, and thus also the alignment requirement per
/// warp</typeparam>
/// <typeparam name="TupelSize">How many elements are merged to tupels in the center part</typeparam>
template <int WarpAlignmentInBytes, int TupelSize> class ThreadSplit
{
  private:
    // store summation of values and not the values itself to reduce number of additions/subtractions needed later
    // during compution
    int mMuted{0};
    int mLeftAndMuted{0};
    int mCenterAndLeftAndMuted{0};
    int mTotal{0};

  public:
    ThreadSplit() = default;

    /// <summary>
    /// Construct ThreadSplit object for a given pointer. Threads will be aligned for optimal computation
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="aPointer"></param>
    /// <param name="aDataWidthInElements"></param>
    template <typename T>
    ThreadSplit(T *aPointer, int aDataWidthInElements)
        requires(TupelSize > 1)
    {
        if (WarpAlignmentInBytes % (to_int(sizeof(T) * TupelSize)) != 0)
        {
            throw INVALIDARGUMENT(aPointer, "Bytes per warp (" << WarpAlignmentInBytes
                                                               << ") is not a multiple of tupel size (" << TupelSize
                                                               << ") times pointer data type size (" << sizeof(T)
                                                               << "). Impossible to fill warp with tupels.");
        }
        if (WarpAlignmentInBytes % (to_int(sizeof(T))) != 0)
        {
            throw INVALIDARGUMENT(WarpAlignmentInBytes, "Bytes per warp ("
                                                            << WarpAlignmentInBytes
                                                            << ") is not a multiple of pointer data type size ("
                                                            << sizeof(T) << "). Impossible to fill warp with tupels.");
        }

        int muted              = GetElementsLeft(aPointer, WarpAlignmentInBytes);
        int left               = GetElementsRight(aPointer, WarpAlignmentInBytes);
        int dataElementsCenter = aDataWidthInElements - left;
        int right              = dataElementsCenter % TupelSize;
        int center             = dataElementsCenter / TupelSize;

        mMuted                 = muted;
        mLeftAndMuted          = muted + left;
        mCenterAndLeftAndMuted = muted + left + center;
        mTotal                 = muted + left + center + right;
    }

    template <typename T>
    ThreadSplit(T *aPointer, int aDataWidthInElements)
        requires(TupelSize == 1)
        : mMuted(0), mLeftAndMuted(0), mCenterAndLeftAndMuted(aDataWidthInElements), mTotal(aDataWidthInElements)
    {
    }
    ~ThreadSplit() = default;

    ThreadSplit(const ThreadSplit &)     = default;
    ThreadSplit(ThreadSplit &&) noexcept = default;

    ThreadSplit &operator=(const ThreadSplit &)     = default;
    ThreadSplit &operator=(ThreadSplit &&) noexcept = default;

    auto operator<=>(const ThreadSplit &) const = default;

    /// <summary>
    /// The number of muted threads
    /// </summary>
    DEVICE_CODE constexpr int Muted() const noexcept
        requires(TupelSize > 1)
    {
        return mMuted;
    }

    /// <summary>
    /// The number of threads in the unaligned left part
    /// </summary>
    DEVICE_CODE constexpr int Left() const noexcept
        requires(TupelSize > 1)
    {
        return mLeftAndMuted - mMuted;
    }

    /// <summary>
    /// The number of threads in the aligned center part
    /// </summary>
    DEVICE_CODE constexpr int Center() const noexcept
        requires(TupelSize > 1)
    {
        return mCenterAndLeftAndMuted - mLeftAndMuted;
    }

    /// <summary>
    /// The number of threads in the unaligned right part
    /// </summary>
    DEVICE_CODE constexpr int Right() const noexcept
        requires(TupelSize > 1)
    {
        return mTotal - mCenterAndLeftAndMuted;
    }

    /// <summary>
    /// The number of muted threads plus number of threads in the unaligned left part
    /// </summary>
    DEVICE_CODE constexpr int MutedAndLeft() const noexcept
        requires(TupelSize > 1)
    {
        return mLeftAndMuted;
    }

    /// <summary>
    /// The number of muted threads plus number of threads in the unaligned left part + number of threads in the aligned
    /// center part
    /// </summary>
    DEVICE_CODE constexpr int MutedAndLeftAndCenter() const noexcept
        requires(TupelSize > 1)
    {
        return mCenterAndLeftAndMuted;
    }

    /// <summary>
    /// Total number of threads
    /// </summary>
    DEVICE_CODE constexpr int Total() const noexcept
        requires(TupelSize > 1)
    {
        return mTotal;
    }

    /// <summary>
    /// Determines if a given thread is outside the defined range, i.e. either muted and &gt;= Total()
    /// </summary>
    DEVICE_CODE constexpr bool ThreadIsOutsideOfRange(int aThreadID) const noexcept
        requires(TupelSize > 1)
    {
        return aThreadID < mMuted || aThreadID >= mTotal;
    }

    /// <summary>
    /// Determines if a given thread is inside the defined range, i.e. not muted and &lt; Total()
    /// </summary>
    DEVICE_CODE constexpr bool ThreadIsInRange(int aThreadID) const noexcept
        requires(TupelSize > 1)
    {
        return aThreadID >= mMuted && aThreadID < mTotal;
    }

    /// <summary>
    /// Determines if a given thread is in the "center"-part, i.e. threads in warp are aligned to WarpAlignmentInBytes
    /// and data can be loaded as tupels
    /// </summary>
    DEVICE_CODE constexpr bool ThreadIsAlignedToWarp(int aThreadID) const noexcept
        requires(TupelSize > 1)
    {
        return aThreadID >= mLeftAndMuted && aThreadID < mCenterAndLeftAndMuted;
    }

    /// <summary>
    /// Translates a threadID to a pixelID
    /// </summary>
    DEVICE_CODE constexpr int GetPixel(int aThreadID) const noexcept
        requires(TupelSize > 1)
    {
        // all threads in a warp will execute the same if clause, so no performance penalty
        if (aThreadID < mLeftAndMuted)
        {
            return aThreadID - mMuted;
        }
        if (aThreadID < mCenterAndLeftAndMuted)
        {
            return Left() + (aThreadID - mLeftAndMuted) * TupelSize;
        }

        // right part
        return Left() + Center() * TupelSize + aThreadID - mCenterAndLeftAndMuted;
    }

    /// <summary>
    /// The number of muted threads
    /// </summary>
    DEVICE_CODE constexpr int Muted() const noexcept
        requires(TupelSize == 1)
    {
        return 0;
    }

    /// <summary>
    /// The number of threads in the unaligned left part
    /// </summary>
    DEVICE_CODE constexpr int Left() const noexcept
        requires(TupelSize == 1)
    {
        return 0;
    }

    /// <summary>
    /// The number of threads in the aligned center part
    /// </summary>
    DEVICE_CODE constexpr int Center() const noexcept
        requires(TupelSize == 1)
    {
        return mTotal;
    }

    /// <summary>
    /// The number of threads in the unaligned right part
    /// </summary>
    DEVICE_CODE constexpr int Right() const noexcept
        requires(TupelSize == 1)
    {
        return 0;
    }

    /// <summary>
    /// The number of muted threads plus number of threads in the unaligned left part
    /// </summary>
    DEVICE_CODE constexpr int MutedAndLeft() const noexcept
        requires(TupelSize == 1)
    {
        return 0;
    }

    /// <summary>
    /// The number of muted threads plus number of threads in the unaligned left part + number of threads in the aligned
    /// center part
    /// </summary>
    DEVICE_CODE constexpr int MutedAndLeftAndCenter() const noexcept
        requires(TupelSize == 1)
    {
        return mTotal;
    }

    /// <summary>
    /// Total number of threads
    /// </summary>
    DEVICE_CODE constexpr int Total() const noexcept
        requires(TupelSize == 1)
    {
        return mTotal;
    }

    /// <summary>
    /// Determines if a given thread is outside the defined range, i.e. either muted and &gt;= Total()
    /// </summary>
    DEVICE_CODE constexpr bool ThreadIsOutsideOfRange(int aThreadID) const noexcept
        requires(TupelSize == 1)
    {
        return aThreadID < 0 || aThreadID >= mTotal;
    }

    /// <summary>
    /// Determines if a given thread is inside the defined range, i.e. not muted and &lt; Total()
    /// </summary>
    DEVICE_CODE constexpr bool ThreadIsInRange(int aThreadID) const noexcept
        requires(TupelSize == 1)
    {
        return aThreadID >= 0 && aThreadID < mTotal;
    }

    /// <summary>
    /// Determines if a given thread is in the "center"-part, i.e. threads in warp are aligned to WarpAlignmentInBytes
    /// and data can be loaded as tupels
    /// </summary>
    DEVICE_CODE constexpr bool ThreadIsAlignedToWarp(int /*aThreadID*/) const noexcept
        requires(TupelSize == 1)
    {
        return true;
    }

    /// <summary>
    /// Translates a threadID to a pixelID
    /// </summary>
    DEVICE_CODE constexpr int GetPixel(int aThreadID) const noexcept
        requires(TupelSize == 1)
    {
        return aThreadID;
    }

  private:
    template <typename T> static ptrdiff_t GetOffsetLeft(T *aPointer, size_t aAlignment = std::alignment_of<T>::value)
    {
        const ptrdiff_t offset = ((ptrdiff_t)aPointer) % aAlignment;
        return -offset;
    }

    template <typename T> static ptrdiff_t GetOffsetRight(T *aPointer, size_t aAlignment = std::alignment_of<T>::value)
    {
        const ptrdiff_t offset = GetOffsetLeft(aPointer, aAlignment) + aAlignment;
        return offset % aAlignment;
    }

    template <typename T> static int GetElementsLeft(T *aPointer, size_t aAlignment = std::alignment_of<T>::value)
    {
        const int offsetBytes = to_int(GetOffsetLeft(aPointer, aAlignment));

        if (offsetBytes % to_int(sizeof(T)) != 0)
        {
            throw INVALIDARGUMENT(aPointer, "The given pointer value " << ptrdiff_t(aPointer)
                                                                       << " is not a multiple of its type size "
                                                                       << sizeof(T) << ".");
        }

        return std::abs(offsetBytes / to_int(sizeof(T)));
    }

    template <typename T> static int GetElementsRight(T *aPointer, size_t aAlignment = std::alignment_of<T>::value)
    {
        const int offsetBytes = to_int(GetOffsetRight(aPointer, aAlignment));

        if (offsetBytes % to_int(sizeof(T)) != 0)
        {
            throw INVALIDARGUMENT(aPointer, "The given pointer value " << ptrdiff_t(aPointer)
                                                                       << " is not a multiple of its type size "
                                                                       << sizeof(T) << ".");
        }

        return offsetBytes / sizeof(T);
    }
};

} // namespace opp::image