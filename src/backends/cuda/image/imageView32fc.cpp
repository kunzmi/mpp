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
template class ImageView<Pixel32fcC1>;
template class ImageView<Pixel32fcC2>;
template class ImageView<Pixel32fcC3>;
template class ImageView<Pixel32fcC4>;
template <>
ImageView<Pixel32fcC1> MPPEXPORT_CUDAI ImageView<Pixel32fcC1>::Null = ImageView<Pixel32fcC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32fcC2> MPPEXPORT_CUDAI ImageView<Pixel32fcC2>::Null = ImageView<Pixel32fcC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32fcC3> MPPEXPORT_CUDAI ImageView<Pixel32fcC3>::Null = ImageView<Pixel32fcC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32fcC4> MPPEXPORT_CUDAI ImageView<Pixel32fcC4>::Null = ImageView<Pixel32fcC4>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertRoundNoAlpha(32fc, 16sc);
ForAllChannelsConvertRoundNoAlpha(32fc, 32sc);

ForAllChannelsConvertRoundScaleNoAlpha(32fc, 16sc);
ForAllChannelsConvertRoundScaleNoAlpha(32fc, 32sc);
InstantiateCopy_For(32fc);

InstantiateSwapChannel_For(32fc);

InstantiateDupNoAlpha_For(32fc);

ForAllChannelsScaleAnyToIntNoAlpha(32fc, 16sc);
ForAllChannelsScaleAnyToIntNoAlpha(32fc, 32sc);
ForAllChannelsScaleAnyToAnyRoundNoAlpha(32fc, 16sc);
ForAllChannelsScaleAnyToAnyRoundNoAlpha(32fc, 32sc);

} // namespace mpp::image::cuda