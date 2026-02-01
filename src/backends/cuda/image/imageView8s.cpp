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
template class ImageView<Pixel8sC1>;
template class ImageView<Pixel8sC2>;
template class ImageView<Pixel8sC3>;
template class ImageView<Pixel8sC4>;
template class ImageView<Pixel8sC4A>;

template <>
ImageView<Pixel8sC1> MPPEXPORT_CUDAI ImageView<Pixel8sC1>::Null = ImageView<Pixel8sC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel8sC2> MPPEXPORT_CUDAI ImageView<Pixel8sC2>::Null = ImageView<Pixel8sC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel8sC3> MPPEXPORT_CUDAI ImageView<Pixel8sC3>::Null = ImageView<Pixel8sC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel8sC4> MPPEXPORT_CUDAI ImageView<Pixel8sC4>::Null = ImageView<Pixel8sC4>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel8sC4A> MPPEXPORT_CUDAI ImageView<Pixel8sC4A>::Null = ImageView<Pixel8sC4A>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertWithAlpha(8s, 8u);
ForAllChannelsConvertWithAlpha(8s, 16u);
ForAllChannelsConvertWithAlpha(8s, 16s);
ForAllChannelsConvertWithAlpha(8s, 32u);
ForAllChannelsConvertWithAlpha(8s, 32s);
ForAllChannelsConvertWithAlpha(8s, 16f);
ForAllChannelsConvertWithAlpha(8s, 16bf);
ForAllChannelsConvertWithAlpha(8s, 32f);
ForAllChannelsConvertWithAlpha(8s, 64f);

InstantiateCopy_For(8s);

InstantiateSwapChannel_For(8s);

InstantiateDup_For(8s);

ForAllChannelsScaleIntToIntWithAlpha(8s, 8u);
ForAllChannelsScaleIntToIntWithAlpha(8s, 16s);
ForAllChannelsScaleIntToIntWithAlpha(8s, 16u);
ForAllChannelsScaleIntToIntWithAlpha(8s, 32s);
ForAllChannelsScaleIntToIntWithAlpha(8s, 32u);

ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 32s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 32u);

ForAllChannelsScaleIntToAnyWithAlpha(8s, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(8s, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(8s, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(8s, 64f);

ForAllChannelsScaleAnyToIntWithAlpha(8s, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(8s, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(8s, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(8s, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(8s, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(8s, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(8s, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(8s, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(8s, 64f);

} // namespace mpp::image::cuda