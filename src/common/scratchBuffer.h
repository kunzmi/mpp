#pragma once
#include <array>
#include <common/defines.h>
#include <common/image/size2D.h>
#include <common/safeCast.h>
#include <concepts>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <vector>

namespace opp
{
struct BufferSize
{
    size_t WidthInElements{0};
    size_t HeightInElements{0};

    BufferSize(int aSizeInElements1D) : WidthInElements(to_size_t(aSizeInElements1D)), HeightInElements(1)
    {
    }
    BufferSize(size_t aSizeInElements1D) : WidthInElements(aSizeInElements1D), HeightInElements(1)
    {
    }
    BufferSize(int aWidthInElements, int aHeightInElements)
        : WidthInElements(to_size_t(aWidthInElements)), HeightInElements(to_size_t(aHeightInElements))
    {
    }
    BufferSize(const opp::image::Size2D &aSizeInElements)
        : WidthInElements(to_size_t(aSizeInElements.x)), HeightInElements(to_size_t(aSizeInElements.y))
    {
    }
};
using BufferSizes = std::vector<BufferSize>;

template <typename T>
concept BufferSizeRange = std::ranges::range<T> && std::same_as<BufferSize, typename std::ranges::range_value_t<T>>;

template <typename... BufferTypes> class ScratchBuffer
{
  private:
    void *mStartPtr;
    std::array<std::size_t, sizeof...(BufferTypes)> mBufferSizesInBytes;
    std::array<std::size_t, sizeof...(BufferTypes)> mLinePitchInBytes;
    static constexpr size_t BUFFER_ALIGNMNENT = 256; // align all temp buffer to 256 bytes

    std::size_t padToAlignment(size_t aValue)
    {
        return ((aValue + (BUFFER_ALIGNMNENT - 1)) / BUFFER_ALIGNMNENT) * BUFFER_ALIGNMNENT;
    }

    void *padToAlignment(void *aValue)
    {
        return reinterpret_cast<void *>(padToAlignment(reinterpret_cast<size_t>(aValue)));
    }

    size_t getOffsetTo(size_t aBufferIndex)
    {
        size_t totalSize = 0;
        for (const auto &elem : mBufferSizesInBytes)
        {
            // size_t sizeOfCurrentBuffer = sizeof();
            totalSize = padToAlignment(totalSize + elem);
        }
        return totalSize;
    }

    template <typename T>
    size_t iterateThroughTupel(size_t index, T /* currentType */, BufferSizeRange auto &aBufferSizesInElements)
    {
        // get buffer sizes in bytes:
        mLinePitchInBytes[index] = aBufferSizesInElements[index].WidthInElements * sizeof(std::remove_pointer_t<T>);
        if (aBufferSizesInElements[index].HeightInElements > 1)
        {
            mLinePitchInBytes[index] = padToAlignment(mLinePitchInBytes[index]);
        }

        const size_t totalSizeInBytes = mLinePitchInBytes[index] * aBufferSizesInElements[index].HeightInElements;

        size_t previousSize = 0;
        if (index > 0)
        {
            previousSize = mBufferSizesInBytes[index - 1];
        }
        size_t currentSize         = previousSize + totalSizeInBytes;
        currentSize                = padToAlignment(currentSize);
        mBufferSizesInBytes[index] = currentSize;

        return index + 1;
    }

  public:
    ScratchBuffer(void *aPtr, BufferSizeRange auto aBufferSizesInElements) : mStartPtr(padToAlignment(aPtr))
    {
        std::tuple<BufferTypes *...> tupelPointerTypes{};
        size_t index = 0;
        std::apply(
            [&](auto &&...pointerTypes) {
                ((index = iterateThroughTupel(index, pointerTypes, aBufferSizesInElements)), ...);
            },
            tupelPointerTypes);
    }
    ~ScratchBuffer() = default;

    ScratchBuffer(const ScratchBuffer &)     = default;
    ScratchBuffer(ScratchBuffer &&) noexcept = default;

    ScratchBuffer &operator=(const ScratchBuffer &)     = default;
    ScratchBuffer &operator=(ScratchBuffer &&) noexcept = default;

    /// <summary>
    /// Return a typed pointer to the i-th sub-buffer
    /// </summary>
    /// <typeparam name="Element"></typeparam>
    /// <returns></returns>
    template <size_t Element>
    std::tuple_element<Element, std::tuple<BufferTypes *...>>::type Get()
        requires(Element < sizeof...(BufferTypes))
    {
        if constexpr (Element > 0)
        {
            return reinterpret_cast<std::tuple_element<Element, std::tuple<BufferTypes *...>>::type>(
                reinterpret_cast<byte *>(mStartPtr) + std::get<Element - 1>(mBufferSizesInBytes));
        }
        else
        {
            return reinterpret_cast<std::tuple_element<Element, std::tuple<BufferTypes *...>>::type>(mStartPtr);
        }
    }

    /// <summary>
    /// returns the size in bytes of the aIndex-th sub buffer including padding.
    /// </summary>
    size_t GetSubBufferSize(size_t aIndex)
    {
        if (aIndex >= sizeof...(BufferTypes))
        {
            return 0;
        }
        if (aIndex == 0)
        {
            return mBufferSizesInBytes[0];
        }
        return mBufferSizesInBytes[aIndex] - mBufferSizesInBytes[aIndex - 1];
    }

    /// <summary>
    /// returns the linepitch in bytes of the aIndex-th sub buffer used for size computation including padding.
    /// </summary>
    size_t GetSubBufferPitch(size_t aIndex)
    {
        if (aIndex >= sizeof...(BufferTypes))
        {
            return 0;
        }
        return mLinePitchInBytes[aIndex];
    }

    /// <summary>
    /// returns the total size in bytes of all sub-buffers including padding.
    /// </summary>
    size_t GetTotalBufferSize()
    {
        // Add BUFFER_ALIGNMNENT to cope with unaligned initial allocations:
        return mBufferSizesInBytes[sizeof...(BufferTypes) - 1] + BUFFER_ALIGNMNENT;
    }
};
} // namespace opp