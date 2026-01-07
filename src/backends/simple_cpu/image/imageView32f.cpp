#include "../dllexport_simplecpu.h" //NOLINT
#define MPPEXPORT_SIMPLECPU32F MPPEXPORT_SIMPLECPU
#include "imageView.h"

namespace mpp::image::cpuSimple
{
#ifndef _WIN32
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32fC1>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32fC2>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32fC3>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32fC4>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32fC4A>;
#endif
} // namespace mpp::image::cpuSimple
#include "imageView_arithmetic_impl.h"                     //NOLINT(misc-include-cleaner)
#include "imageView_colorConversion_impl.h"                //NOLINT(misc-include-cleaner)
#include "imageView_dataExchangeAndInit_impl.h"            //NOLINT(misc-include-cleaner)
#include "imageView_filtering_impl.h"                      //NOLINT(misc-include-cleaner)
#include "imageView_geometryTransforms_affine_impl.h"      //NOLINT(misc-include-cleaner)
#include "imageView_geometryTransforms_impl.h"             //NOLINT(misc-include-cleaner)
#include "imageView_geometryTransforms_perspective_impl.h" //NOLINT(misc-include-cleaner)
#include "imageView_geometryTransforms_resize_impl.h"      //NOLINT(misc-include-cleaner)
#include "imageView_geometryTransforms_rotate_impl.h"      //NOLINT(misc-include-cleaner)
#include "imageView_morphology_impl.h"                     //NOLINT(misc-include-cleaner)
#include "imageView_statistics_impl.h"                     //NOLINT(misc-include-cleaner)
#include "imageView_thresholdAndCompare_impl.h"            //NOLINT(misc-include-cleaner)
#include "instantiateConversion.h"
#include "instantiateCopy.h"
#include "instantiateDup.h"
#include "instantiateScale.h"
#include "instantiateSwapChannel.h"
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/forEachPixelMasked_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixelPlanar.h>
#include <backends/simple_cpu/image/forEachPixelPlanar_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixelSingleChannel.h>
#include <backends/simple_cpu/image/forEachPixelSingleChannel_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixel_impl.h>              //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/reduction.h>
#include <backends/simple_cpu/image/reductionMasked.h>
#include <backends/simple_cpu/image/reductionMasked_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/reduction_impl.h>       //NOLINT(misc-include-cleaner)
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h> //NOLINT(misc-include-cleaner)

namespace mpp::image::cpuSimple
{

template class ImageView<Pixel32fC1>;
template class ImageView<Pixel32fC2>;
template class ImageView<Pixel32fC3>;
template class ImageView<Pixel32fC4>;
template class ImageView<Pixel32fC4A>;

ForAllChannelsConvertWithAlpha(32f, 16f);
ForAllChannelsConvertWithAlpha(32f, 16bf);
ForAllChannelsConvertWithAlpha(32f, 64f);

ForAllChannelsConvertRoundWithAlpha(32f, 8u);
ForAllChannelsConvertRoundWithAlpha(32f, 8s);
ForAllChannelsConvertRoundWithAlpha(32f, 16u);
ForAllChannelsConvertRoundWithAlpha(32f, 16s);
ForAllChannelsConvertRoundWithAlpha(32f, 32u);
ForAllChannelsConvertRoundWithAlpha(32f, 32s);
ForAllChannelsConvertRoundWithAlpha(32f, 16bf);
ForAllChannelsConvertRoundWithAlpha(32f, 16f);

ForAllChannelsConvertRoundScaleWithAlpha(32f, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 16s);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 16u);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 32s);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 32u);

InstantiateCopy_For(32f);

InstantiateSwapChannel_For(32f);

InstantiateDup_For(32f);

ForAllChannelsScaleAnyToIntWithAlpha(32f, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(32f, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(32f, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(32f, 64f);

} // namespace mpp::image::cpuSimple