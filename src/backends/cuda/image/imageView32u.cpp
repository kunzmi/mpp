#include "dllexport_cudai.h" //NOLINT(misc-include-cleaner)

#include "dataExchangeAndInit/instantiateConversion.h"
#include "dataExchangeAndInit/instantiateCopy.h"
#include "dataExchangeAndInit/instantiateDup.h"
#include "dataExchangeAndInit/instantiateScale.h"
#include "dataExchangeAndInit/instantiateSwapChannel.h"
#include "imageView.h"
#include "imageView_arithmetic_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_colorConversion_impl.h"     //NOLINT(misc-include-cleaner)
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
template class ImageView<Pixel32uC1>;
template class ImageView<Pixel32uC2>;
template class ImageView<Pixel32uC3>;
template class ImageView<Pixel32uC4>;
template class ImageView<Pixel32uC4A>;
template <>
ImageView<Pixel32uC1> MPPEXPORT_CUDAI ImageView<Pixel32uC1>::Null = ImageView<Pixel32uC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32uC2> MPPEXPORT_CUDAI ImageView<Pixel32uC2>::Null = ImageView<Pixel32uC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32uC3> MPPEXPORT_CUDAI ImageView<Pixel32uC3>::Null = ImageView<Pixel32uC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32uC4> MPPEXPORT_CUDAI ImageView<Pixel32uC4>::Null = ImageView<Pixel32uC4>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32uC4A> MPPEXPORT_CUDAI ImageView<Pixel32uC4A>::Null = ImageView<Pixel32uC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertWithAlpha(32u, 8u);
ForAllChannelsConvertWithAlpha(32u, 8s);
ForAllChannelsConvertWithAlpha(32u, 16u);
ForAllChannelsConvertWithAlpha(32u, 16s);
ForAllChannelsConvertWithAlpha(32u, 32s);
ForAllChannelsConvertWithAlpha(32u, 16bf);
ForAllChannelsConvertWithAlpha(32u, 16f);
ForAllChannelsConvertWithAlpha(32u, 32f);
ForAllChannelsConvertWithAlpha(32u, 64f);

ForAllChannelsConvertRoundScaleWithAlpha(32u, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(32u, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(32u, 16s);
ForAllChannelsConvertRoundScaleWithAlpha(32u, 16u);
ForAllChannelsConvertRoundScaleWithAlpha(32u, 32s);

InstantiateCopy_For(32u);

InstantiateSwapChannel_For(32u);

InstantiateDup_For(32u);

ForAllChannelsScaleIntToIntWithAlpha(32u, 8s);
ForAllChannelsScaleIntToIntWithAlpha(32u, 8u);
ForAllChannelsScaleIntToIntWithAlpha(32u, 16s);
ForAllChannelsScaleIntToIntWithAlpha(32u, 16u);
ForAllChannelsScaleIntToIntWithAlpha(32u, 32s);

ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 32s);
ForAllChannelsScaleIntToAnyWithAlpha(32u, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(32u, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(32u, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(32u, 64f);

ForAllChannelsScaleAnyToIntWithAlpha(32u, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(32u, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(32u, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(32u, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(32u, 32s);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 32s);
ForAllChannelsScaleAnyToAnyWithAlpha(32u, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(32u, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(32u, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(32u, 64f);

} // namespace mpp::image::cuda