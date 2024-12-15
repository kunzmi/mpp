#include "image8uC1View.h"
#include "imageView.h"
#include <common/image/border.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>

namespace opp::image::npp
{
Image8uC1View::Image8uC1View(const Size2D &aSize) : ImageView<Pixel8uC1>(aSize)
{
}

Image8uC1View::Image8uC1View(Pixel8uC1 *aBasePointer, const SizePitched &aSizeAlloc)
    : ImageView<Pixel8uC1>(aBasePointer, aSizeAlloc)
{
}
Image8uC1View::Image8uC1View(Pixel8uC1 *aBasePointer, const SizePitched &aSizeAlloc, const Roi &aRoi)
    : ImageView<Pixel8uC1>(aBasePointer, aSizeAlloc, aRoi)
{
}

Image8uC1View Image8uC1View::GetView(const Roi &aRoi)
{
    return {Pointer(), SizePitched(SizeAlloc(), Pitch()), aRoi};
}

Image8uC1View Image8uC1View::GetView(const Border &aBorder)
{
    const Roi newRoi = ROI() + aBorder;
    checkRoiIsInRoi(newRoi, Roi(0, 0, SizeAlloc()));
    return {Pointer(), SizePitched(SizeAlloc(), Pitch()), newRoi};
}

} // namespace opp::image::npp