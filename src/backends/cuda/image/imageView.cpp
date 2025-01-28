#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "imageView.h"
#include "imageView_arithmetic_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_dataExchangeAndInit_impl.h" //NOLINT(misc-include-cleaner)
#include <common/image/pixelTypes.h>

namespace opp::image::cuda
{
template class ImageView<Pixel8uC1>;
template class ImageView<Pixel8uC2>;
template class ImageView<Pixel8uC3>;
template class ImageView<Pixel8uC4>;
template class ImageView<Pixel8uC4A>;

using Image8uC1View  = ImageView<Pixel8uC1>;
using Image8uC2View  = ImageView<Pixel8uC2>;
using Image8uC3View  = ImageView<Pixel8uC3>;
using Image8uC4View  = ImageView<Pixel8uC4>;
using Image8uC4AView = ImageView<Pixel8uC4A>;

template class ImageView<Pixel32fC1>;
template class ImageView<Pixel32fC2>;
template class ImageView<Pixel32fC3>;
template class ImageView<Pixel32fC4>;
template class ImageView<Pixel32fC4A>;

using Image32fC1View  = ImageView<Pixel32fC1>;
using Image32fC2View  = ImageView<Pixel32fC2>;
using Image32fC3View  = ImageView<Pixel32fC3>;
using Image32fC4View  = ImageView<Pixel32fC4>;
using Image32fC4AView = ImageView<Pixel32fC4A>;

template class ImageView<Pixel32fcC1>;
using Image32fcC1View = ImageView<Pixel32fcC1>;
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND