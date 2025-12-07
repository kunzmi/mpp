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
template class ImageView<Pixel32fC1>;
template class ImageView<Pixel32fC2>;
template class ImageView<Pixel32fC3>;
template class ImageView<Pixel32fC4>;
template class ImageView<Pixel32fC4A>;
template <> ImageView<Pixel32fC1> ImageView<Pixel32fC1>::Null   = ImageView<Pixel32fC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC2> ImageView<Pixel32fC2>::Null   = ImageView<Pixel32fC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC3> ImageView<Pixel32fC3>::Null   = ImageView<Pixel32fC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC4> ImageView<Pixel32fC4>::Null   = ImageView<Pixel32fC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC4A> ImageView<Pixel32fC4A>::Null = ImageView<Pixel32fC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertWithAlpha(32f, 16f);
ForAllChannelsConvertWithAlpha(32f, 16bf);
ForAllChannelsConvertWithAlpha(32f, 64f);

ForAllChannelsConvertRoundWithAlpha(32f, 8u);
ForAllChannelsConvertRoundWithAlpha(32f, 8s);
ForAllChannelsConvertRoundWithAlpha(32f, 16u);
ForAllChannelsConvertRoundWithAlpha(32f, 16s);
ForAllChannelsConvertRoundWithAlpha(32f, 16bf);
ForAllChannelsConvertRoundWithAlpha(32f, 16f);

ForAllChannelsConvertRoundScaleWithAlpha(32f, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 16s);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 16u);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 32s);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 32u);

InstantiateCopy_For(32f);

InstantiateSwapChannel_For(32f);

InstantiateDup_For(32f);

ForAllChannelsScaleAnyToIntWithAlpha(32f, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(32f, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(32f, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(32f, 64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND