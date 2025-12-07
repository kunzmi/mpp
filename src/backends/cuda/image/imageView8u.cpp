#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND
#include "dllexport_cudai.h" //NOLINT(misc-include-cleaner)

#include "dataExchangeAndInit/instantiateConversion.h"
#include "dataExchangeAndInit/instantiateCopy.h"
#include "dataExchangeAndInit/instantiateDup.h"
#include "dataExchangeAndInit/instantiateScale.h"
#include "dataExchangeAndInit/instantiateSwapChannel.h"
#include "imageView.h"
#include "imageView_arithmetic_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_dataExchangeAndInit_impl.h" //NOLINT(misc-include-cleaner)
#include "imageView_filtering_impl.h"           //NOLINT(misc-include-cleaner)
#include "imageView_geometryTransforms_impl.h"  //NOLINT(misc-include-cleaner)
#include "imageView_morphology_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_statistics_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_thresholdAndCompare_impl.h" //NOLINT(misc-include-cleaner)
#include <backends/cuda/streamCtx.h>            //NOLINT(misc-include-cleaner)
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h> //NOLINT(misc-include-cleaner)

namespace mpp::image::cuda
{
template class ImageView<Pixel8uC1>;
template class ImageView<Pixel8uC2>;
template class ImageView<Pixel8uC3>;
template class ImageView<Pixel8uC4>;
template class ImageView<Pixel8uC4A>;

template <> ImageView<Pixel8uC1> ImageView<Pixel8uC1>::Null   = ImageView<Pixel8uC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8uC2> ImageView<Pixel8uC2>::Null   = ImageView<Pixel8uC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8uC3> ImageView<Pixel8uC3>::Null   = ImageView<Pixel8uC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8uC4> ImageView<Pixel8uC4>::Null   = ImageView<Pixel8uC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8uC4A> ImageView<Pixel8uC4A>::Null = ImageView<Pixel8uC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertWithAlpha(8u, 8s);
ForAllChannelsConvertWithAlpha(8u, 16s);
ForAllChannelsConvertWithAlpha(8u, 16u);
ForAllChannelsConvertWithAlpha(8u, 32s);
ForAllChannelsConvertWithAlpha(8u, 32u);
ForAllChannelsConvertWithAlpha(8u, 16f);
ForAllChannelsConvertWithAlpha(8u, 16bf);
ForAllChannelsConvertWithAlpha(8u, 32f);
ForAllChannelsConvertWithAlpha(8u, 64f);

ForAllChannelsConvertRoundScaleWithAlpha(8u, 8s);

InstantiateCopy_For(8u);

InstantiateSwapChannel_For(8u);

InstantiateDup_For(8u);

ForAllChannelsScaleIntToIntWithAlpha(8u, 8s);
ForAllChannelsScaleIntToIntWithAlpha(8u, 16s);
ForAllChannelsScaleIntToIntWithAlpha(8u, 16u);
ForAllChannelsScaleIntToIntWithAlpha(8u, 32s);
ForAllChannelsScaleIntToIntWithAlpha(8u, 32u);

ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 32s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 64f);

ForAllChannelsScaleAnyToIntWithAlpha(8u, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND