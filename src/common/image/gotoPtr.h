#pragma once
#include <common/defines.h>
#include <cstddef>

namespace opp::image
{
// integer indices
template <typename T> DEVICE_CODE T *gotoPtr(T *aPixel0, size_t aPitch, int x, int y)
{
    return (((T *)((char *)aPixel0 + aPitch * size_t(y))) + size_t(x));
}
template <typename T> DEVICE_CODE const T *gotoPtr(const T *aPixel0, size_t aPitch, int x, int y)
{
    return (((const T *)((const char *)aPixel0 + aPitch * size_t(y))) + size_t(x));
}

template <ByteSizeType T> DEVICE_CODE T *gotoPtr(T *aPixel0, size_t aPitch, int x, int y)
{
    return ((aPixel0 + aPitch * size_t(y)) + size_t(x));
}

template <ByteSizeType T> DEVICE_CODE const T *gotoPtr(const T *aPixel0, size_t aPitch, int x, int y)
{
    return ((aPixel0 + aPitch * size_t(y)) + size_t(x));
}

// size_t indices
template <typename T> DEVICE_CODE T *gotoPtr(T *aPixel0, size_t aPitch, size_t x, size_t y)
{
    return (((T *)((char *)aPixel0 + aPitch * y)) + x);
}
template <typename T> DEVICE_CODE const T *gotoPtr(const T *aPixel0, size_t aPitch, size_t x, size_t y)
{
    return (((const T *)((const char *)aPixel0 + aPitch * y)) + x);
}

template <ByteSizeType T> DEVICE_CODE T *gotoPtr(T *aPixel0, size_t aPitch, size_t x, size_t y)
{
    return ((aPixel0 + aPitch * y) + x);
}

template <ByteSizeType T> DEVICE_CODE const T *gotoPtr(const T *aPixel0, size_t aPitch, size_t x, size_t y)
{
    return ((aPixel0 + aPitch * y) + x);
}
} // namespace opp::image