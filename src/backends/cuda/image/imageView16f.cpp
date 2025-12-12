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
template class ImageView<Pixel16fC1>;
template class ImageView<Pixel16fC2>;
template class ImageView<Pixel16fC3>;
template class ImageView<Pixel16fC4>;
template class ImageView<Pixel16fC4A>;
template <>
ImageView<Pixel16fC1> MPPEXPORT_CUDAI ImageView<Pixel16fC1>::Null = ImageView<Pixel16fC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16fC2> MPPEXPORT_CUDAI ImageView<Pixel16fC2>::Null = ImageView<Pixel16fC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16fC3> MPPEXPORT_CUDAI ImageView<Pixel16fC3>::Null = ImageView<Pixel16fC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16fC4> MPPEXPORT_CUDAI ImageView<Pixel16fC4>::Null = ImageView<Pixel16fC4>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16fC4A> MPPEXPORT_CUDAI ImageView<Pixel16fC4A>::Null = ImageView<Pixel16fC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertRoundWithAlpha(16f, 8u);
ForAllChannelsConvertRoundWithAlpha(16f, 8s);
ForAllChannelsConvertRoundWithAlpha(16f, 16u);
ForAllChannelsConvertRoundWithAlpha(16f, 16s);
ForAllChannelsConvertRoundWithAlpha(16f, 32u);
ForAllChannelsConvertRoundWithAlpha(16f, 32s);

InstantiateCopy_For(16f);

InstantiateSwapChannel_For(16f);

InstantiateDup_For(16f);

ForAllChannelsScaleAnyToIntWithAlpha(16f, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(16f, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(16f, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(16f, 64f);

} // namespace mpp::image::cuda