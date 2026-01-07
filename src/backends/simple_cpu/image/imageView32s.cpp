#include "../dllexport_simplecpu.h" //NOLINT
#define MPPEXPORT_SIMPLECPU32S MPPEXPORT_SIMPLECPU
#include "imageView.h"
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
#ifndef _WIN32
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32sC1>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32sC2>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32sC3>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32sC4>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32sC4A>;
#endif

template class ImageView<Pixel32sC1>;
template class ImageView<Pixel32sC2>;
template class ImageView<Pixel32sC3>;
template class ImageView<Pixel32sC4>;
template class ImageView<Pixel32sC4A>;

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

} // namespace mpp::image::cpuSimple