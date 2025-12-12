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
template class ImageView<Pixel64fC1>;
template class ImageView<Pixel64fC2>;
template class ImageView<Pixel64fC3>;
template class ImageView<Pixel64fC4>;
template class ImageView<Pixel64fC4A>;
template <>
ImageView<Pixel64fC1> MPPEXPORT_CUDAI ImageView<Pixel64fC1>::Null = ImageView<Pixel64fC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel64fC2> MPPEXPORT_CUDAI ImageView<Pixel64fC2>::Null = ImageView<Pixel64fC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel64fC3> MPPEXPORT_CUDAI ImageView<Pixel64fC3>::Null = ImageView<Pixel64fC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel64fC4> MPPEXPORT_CUDAI ImageView<Pixel64fC4>::Null = ImageView<Pixel64fC4>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel64fC4A> MPPEXPORT_CUDAI ImageView<Pixel64fC4A>::Null = ImageView<Pixel64fC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertWithAlpha(64f, 16f);
ForAllChannelsConvertWithAlpha(64f, 16bf);
ForAllChannelsConvertWithAlpha(64f, 32f);

ForAllChannelsConvertRoundWithAlpha(64f, 8u);
ForAllChannelsConvertRoundWithAlpha(64f, 8s);
ForAllChannelsConvertRoundWithAlpha(64f, 16u);
ForAllChannelsConvertRoundWithAlpha(64f, 16s);
ForAllChannelsConvertRoundWithAlpha(64f, 32u);
ForAllChannelsConvertRoundWithAlpha(64f, 32s);

ForAllChannelsConvertRoundScaleWithAlpha(64f, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 16s);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 16u);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 32s);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 32u);

InstantiateCopy_For(64f);

InstantiateSwapChannel_For(64f);

InstantiateDup_For(64f);

ForAllChannelsScaleAnyToIntWithAlpha(64f, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(64f, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(64f, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(64f, 32f);

} // namespace mpp::image::cuda