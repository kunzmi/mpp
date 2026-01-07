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
template class ImageView<Pixel16bfC1>;
template class ImageView<Pixel16bfC2>;
template class ImageView<Pixel16bfC3>;
template class ImageView<Pixel16bfC4>;
template class ImageView<Pixel16bfC4A>;
template <>
ImageView<Pixel16bfC1> MPPEXPORT_CUDAI ImageView<Pixel16bfC1>::Null = ImageView<Pixel16bfC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16bfC2> MPPEXPORT_CUDAI ImageView<Pixel16bfC2>::Null = ImageView<Pixel16bfC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16bfC3> MPPEXPORT_CUDAI ImageView<Pixel16bfC3>::Null = ImageView<Pixel16bfC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16bfC4> MPPEXPORT_CUDAI ImageView<Pixel16bfC4>::Null = ImageView<Pixel16bfC4>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel16bfC4A> MPPEXPORT_CUDAI ImageView<Pixel16bfC4A>::Null =
    ImageView<Pixel16bfC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertWithAlpha(16bf, 32f);
ForAllChannelsConvertRoundWithAlpha(16bf, 8u);
ForAllChannelsConvertRoundWithAlpha(16bf, 8s);
ForAllChannelsConvertRoundWithAlpha(16bf, 16u);
ForAllChannelsConvertRoundWithAlpha(16bf, 16s);
ForAllChannelsConvertRoundWithAlpha(16bf, 32u);
ForAllChannelsConvertRoundWithAlpha(16bf, 32s);

InstantiateCopy_For(16bf);

InstantiateSwapChannel_For(16bf);

InstantiateDup_For(16bf);

ForAllChannelsScaleAnyToIntWithAlpha(16bf, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(16bf, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(16bf, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(16bf, 64f);

} // namespace mpp::image::cuda