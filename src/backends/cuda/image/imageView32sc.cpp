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
template class ImageView<Pixel32scC1>;
template class ImageView<Pixel32scC2>;
template class ImageView<Pixel32scC3>;
template class ImageView<Pixel32scC4>;
template <>
ImageView<Pixel32scC1> MPPEXPORT_CUDAI ImageView<Pixel32scC1>::Null = ImageView<Pixel32scC1>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32scC2> MPPEXPORT_CUDAI ImageView<Pixel32scC2>::Null = ImageView<Pixel32scC2>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32scC3> MPPEXPORT_CUDAI ImageView<Pixel32scC3>::Null = ImageView<Pixel32scC3>(nullptr, Size2D(0, 0), 0);
template <>
ImageView<Pixel32scC4> MPPEXPORT_CUDAI ImageView<Pixel32scC4>::Null = ImageView<Pixel32scC4>(nullptr, Size2D(0, 0), 0);

ForAllChannelsConvertNoAlpha(32sc, 16sc);
ForAllChannelsConvertNoAlpha(32sc, 32fc);

ForAllChannelsConvertRoundScaleNoAlpha(32sc, 16sc);

InstantiateCopy_For(32sc);

InstantiateSwapChannel_For(32sc);

InstantiateDupNoAlpha_For(32sc);

ForAllChannelsScaleIntToIntNoAlpha(32sc, 16sc);

ForAllChannelsScaleIntToAnyRoundNoAlpha(32sc, 16sc);
ForAllChannelsScaleIntToAnyNoAlpha(32sc, 32fc);

ForAllChannelsScaleAnyToIntNoAlpha(32sc, 16sc);

ForAllChannelsScaleAnyToAnyRoundNoAlpha(32sc, 16sc);
ForAllChannelsScaleAnyToAnyNoAlpha(32sc, 32fc);

} // namespace mpp::image::cuda