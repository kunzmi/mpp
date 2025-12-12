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
template class ImageView<Pixel16sC1>;
template class ImageView<Pixel16sC2>;
template class ImageView<Pixel16sC3>;
template class ImageView<Pixel16sC4>;
template class ImageView<Pixel16sC4A>;
template <>
ImageView<Pixel16sC1> MPPEXPORT_CUDAI ImageView<Pixel16sC1>::Null = ImageView<Pixel16sC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16sC2> MPPEXPORT_CUDAI ImageView<Pixel16sC2>::Null = ImageView<Pixel16sC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16sC3> MPPEXPORT_CUDAI ImageView<Pixel16sC3>::Null = ImageView<Pixel16sC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16sC4> MPPEXPORT_CUDAI ImageView<Pixel16sC4>::Null = ImageView<Pixel16sC4>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16sC4A> MPPEXPORT_CUDAI ImageView<Pixel16sC4A>::Null = ImageView<Pixel16sC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertWithAlpha(16s, 8u);
ForAllChannelsConvertWithAlpha(16s, 8s);
ForAllChannelsConvertWithAlpha(16s, 16u);
ForAllChannelsConvertWithAlpha(16s, 32s);
ForAllChannelsConvertWithAlpha(16s, 32u);
ForAllChannelsConvertWithAlpha(16s, 16f);
ForAllChannelsConvertWithAlpha(16s, 16bf);
ForAllChannelsConvertWithAlpha(16s, 32f);
ForAllChannelsConvertWithAlpha(16s, 64f);

ForAllChannelsConvertRoundScaleWithAlpha(16s, 8s);

InstantiateCopy_For(16s);

InstantiateSwapChannel_For(16s);

InstantiateDup_For(16s);

ForAllChannelsScaleIntToIntWithAlpha(16s, 8u);
ForAllChannelsScaleIntToIntWithAlpha(16s, 8s);
ForAllChannelsScaleIntToIntWithAlpha(16s, 16u);
ForAllChannelsScaleIntToIntWithAlpha(16s, 32s);
ForAllChannelsScaleIntToIntWithAlpha(16s, 32u);

ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 32s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(16s, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(16s, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(16s, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(16s, 64f);

ForAllChannelsScaleAnyToIntWithAlpha(16s, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(16s, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(16s, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(16s, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(16s, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(16s, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(16s, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(16s, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(16s, 64f);

} // namespace mpp::image::cuda