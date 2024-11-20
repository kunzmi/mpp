#include "size2D.h"
#include "sizePitched.h"
#include <cstddef>

namespace opp::image
{

SizePitched::SizePitched(const Size2D &aSize, size_t aPitch) noexcept : mSize(aSize), mPitch(aPitch)
{
}

const Size2D &SizePitched::Size() const
{
    return mSize;
}

Size2D &SizePitched::Size()
{
    return mSize;
}

size_t SizePitched::Pitch() const
{
    return mPitch;
}

} // namespace opp::image