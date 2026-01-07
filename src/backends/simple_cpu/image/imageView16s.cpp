#include "../dllexport_simplecpu.h" //NOLINT
#define MPPEXPORT_SIMPLECPU16S MPPEXPORT_SIMPLECPU
#include "imageView.h"

namespace mpp::image::cpuSimple
{
#ifndef _WIN32
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel16sC1>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel16sC2>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel16sC3>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel16sC4>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel16sC4A>;
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

template class ImageView<Pixel16sC1>;
template class ImageView<Pixel16sC2>;
template class ImageView<Pixel16sC3>;
template class ImageView<Pixel16sC4>;
template class ImageView<Pixel16sC4A>;

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

} // namespace mpp::image::cpuSimple