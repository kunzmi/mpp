#pragma once
#include "size2D.h"
#include <common/defines.h>
#include <cstddef>

namespace opp::image
{
/// <summary>
/// A specialized type to describe size in 2D (number of pixels in X and Y) with pitch information
/// </summary>
class SizePitched
{
  public:
    SizePitched(const Size2D &aSize, size_t aPitch) noexcept;

    ~SizePitched() = default;

    SizePitched(const SizePitched &)                     = default;
    SizePitched(SizePitched &&)                          = default;
    SizePitched &operator=(const SizePitched &) noexcept = default;
    SizePitched &operator=(SizePitched &&) noexcept      = default;

    [[nodiscard]] const Size2D &Size() const;
    [[nodiscard]] Size2D &Size();
    [[nodiscard]] size_t Pitch() const;

  private:
    Size2D mSize;
    size_t mPitch;
};
} // namespace opp::image