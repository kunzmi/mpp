#pragma once
#include <array>
#include <common/defines.h>
#include <concepts>
#include <ranges>
#include <tuple>

namespace opp
{
template <typename T>
concept SizeTRange = std::ranges::range<T> && std::same_as<size_t, typename std::ranges::range_value_t<T>>;

template <typename... BufferTypes> class ScratchBuffer
{
  private:
    void *mStartPtr;
    std::array<std::size_t, sizeof...(BufferTypes)> mBufferSizesInBytes;
    static constexpr size_t BUFFER_ALIGNMNENT = 256; // align all temp buffer to 256 bytes

    std::size_t padToAlignment(size_t aValue)
    {
        return ((aValue + (BUFFER_ALIGNMNENT - 1)) / BUFFER_ALIGNMNENT) * BUFFER_ALIGNMNENT;
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
    size_t iterateThroughTupel(size_t index, T /* currentType */, SizeTRange auto &aBufferSizesInElements)
    {
        size_t previousSize = 0;
        if (index > 0)
        {
            previousSize = mBufferSizesInBytes[index - 1];
        }
        size_t currentSize         = previousSize + aBufferSizesInElements[index] * sizeof(std::remove_pointer_t<T>);
        currentSize                = padToAlignment(currentSize);
        mBufferSizesInBytes[index] = currentSize;

        return index + 1;
    }

  public:
    ScratchBuffer(void *aPtr, SizeTRange auto aBufferSizesInElements) : mStartPtr(aPtr)
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
    /// returns the total size in bytes of all sub-buffers including padding.
    /// </summary>
    size_t GetTotalBufferSize()
    {
        return mBufferSizesInBytes[sizeof...(BufferTypes) - 1];
    }
};
} // namespace opp