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
template class ImageView<Pixel16uC1>;
template class ImageView<Pixel16uC2>;
template class ImageView<Pixel16uC3>;
template class ImageView<Pixel16uC4>;
template class ImageView<Pixel16uC4A>;

template <>
ImageView<Pixel16uC1> MPPEXPORT_CUDAI ImageView<Pixel16uC1>::Null = ImageView<Pixel16uC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16uC2> MPPEXPORT_CUDAI ImageView<Pixel16uC2>::Null = ImageView<Pixel16uC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16uC3> MPPEXPORT_CUDAI ImageView<Pixel16uC3>::Null = ImageView<Pixel16uC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16uC4> MPPEXPORT_CUDAI ImageView<Pixel16uC4>::Null = ImageView<Pixel16uC4>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16uC4A> MPPEXPORT_CUDAI ImageView<Pixel16uC4A>::Null = ImageView<Pixel16uC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertWithAlpha(16u, 8u);
ForAllChannelsConvertWithAlpha(16u, 8s);
ForAllChannelsConvertWithAlpha(16u, 16s);
ForAllChannelsConvertWithAlpha(16u, 32s);
ForAllChannelsConvertWithAlpha(16u, 32u);
ForAllChannelsConvertWithAlpha(16u, 16f);
ForAllChannelsConvertWithAlpha(16u, 16bf);
ForAllChannelsConvertWithAlpha(16u, 32f);
ForAllChannelsConvertWithAlpha(16u, 64f);

ForAllChannelsConvertRoundScaleWithAlpha(16u, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(16u, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(16u, 16s);

InstantiateCopy_For(16u);

InstantiateSwapChannel_For(16u);

InstantiateDup_For(16u);

ForAllChannelsScaleIntToIntWithAlpha(16u, 8s);
ForAllChannelsScaleIntToIntWithAlpha(16u, 8u);
ForAllChannelsScaleIntToIntWithAlpha(16u, 16s);
ForAllChannelsScaleIntToIntWithAlpha(16u, 32s);
ForAllChannelsScaleIntToIntWithAlpha(16u, 32u);

ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 32s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(16u, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(16u, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(16u, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(16u, 64f);

ForAllChannelsScaleAnyToIntWithAlpha(16u, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(16u, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(16u, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(16u, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(16u, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(16u, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(16u, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(16u, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(16u, 64f);

} // namespace mpp::image::cuda