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
template class ImageView<Pixel32sC1>;
template class ImageView<Pixel32sC2>;
template class ImageView<Pixel32sC3>;
template class ImageView<Pixel32sC4>;
template class ImageView<Pixel32sC4A>;
template <>
ImageView<Pixel32sC1> MPPEXPORT_CUDAI ImageView<Pixel32sC1>::Null = ImageView<Pixel32sC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32sC2> MPPEXPORT_CUDAI ImageView<Pixel32sC2>::Null = ImageView<Pixel32sC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32sC3> MPPEXPORT_CUDAI ImageView<Pixel32sC3>::Null = ImageView<Pixel32sC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32sC4> MPPEXPORT_CUDAI ImageView<Pixel32sC4>::Null = ImageView<Pixel32sC4>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32sC4A> MPPEXPORT_CUDAI ImageView<Pixel32sC4A>::Null = ImageView<Pixel32sC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertWithAlpha(32s, 8u);
ForAllChannelsConvertWithAlpha(32s, 8s);
ForAllChannelsConvertWithAlpha(32s, 16u);
ForAllChannelsConvertWithAlpha(32s, 16s);
ForAllChannelsConvertWithAlpha(32s, 32u);
ForAllChannelsConvertWithAlpha(32s, 16bf);
ForAllChannelsConvertWithAlpha(32s, 16f);
ForAllChannelsConvertWithAlpha(32s, 32f);
ForAllChannelsConvertWithAlpha(32s, 64f);

ForAllChannelsConvertRoundScaleWithAlpha(32s, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(32s, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(32s, 16s);
ForAllChannelsConvertRoundScaleWithAlpha(32s, 16u);

InstantiateCopy_For(32s);

InstantiateSwapChannel_For(32s);

InstantiateDup_For(32s);

ForAllChannelsScaleIntToIntWithAlpha(32s, 8u);
ForAllChannelsScaleIntToIntWithAlpha(32s, 8s);
ForAllChannelsScaleIntToIntWithAlpha(32s, 16u);
ForAllChannelsScaleIntToIntWithAlpha(32s, 16s);
ForAllChannelsScaleIntToIntWithAlpha(32s, 32u);

ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(32s, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(32s, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(32s, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(32s, 64f);

ForAllChannelsScaleAnyToIntWithAlpha(32s, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(32s, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(32s, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(32s, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(32s, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(32s, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(32s, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(32s, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(32s, 64f);

} // namespace mpp::image::cuda