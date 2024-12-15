#pragma once

#include "imageView.h"
#include <backends/npp/nppException.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <vector>

namespace opp::image::npp
{
class Image8uC1View : public ImageView<Pixel8uC1>
{
  protected:
    Image8uC1View() = default;
    explicit Image8uC1View(const Size2D &aSize);

  public:
    Image8uC1View(Pixel8uC1 *aBasePointer, const SizePitched &aSizeAlloc);
    Image8uC1View(Pixel8uC1 *aBasePointer, const SizePitched &aSizeAlloc, const Roi &aRoi);
    ~Image8uC1View() = default;

    Image8uC1View(const Image8uC1View &)     = default;
    Image8uC1View(Image8uC1View &&) noexcept = default;

    Image8uC1View &operator=(const Image8uC1View &)     = default;
    Image8uC1View &operator=(Image8uC1View &&) noexcept = default;

    /// <summary>
    /// Returns a new Image8uC1View with the new ROI
    /// </summary>
    Image8uC1View GetView(const Roi &aRoi);

    /// <summary>
    /// Returns a new ImageView with the current ROI adapted by aBorder
    /// </summary>
    Image8uC1View GetView(const Border &aBorder = Border());
};
} // namespace opp::image::npp