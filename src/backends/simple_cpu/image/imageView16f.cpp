#include "imageView.h"
#include "imageView_arithmetic_impl.h"                     //NOLINT(misc-include-cleaner)
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
#include <backends/simple_cpu/image/forEachPixel_impl.h>              //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/forEachPixelMasked_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixelPlanar.h>
#include <backends/simple_cpu/image/forEachPixelPlanar_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixelSingleChannel.h>
#include <backends/simple_cpu/image/forEachPixelSingleChannel_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/reduction.h>
#include <backends/simple_cpu/image/reduction_impl.h>       //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/reductionMasked.h>
#include <backends/simple_cpu/image/reductionMasked_impl.h> //NOLINT(misc-include-cleaner)
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h> //NOLINT(misc-include-cleaner)

namespace mpp::image::cpuSimple
{
template class ImageView<Pixel16fC1>;
template class ImageView<Pixel16fC2>;
template class ImageView<Pixel16fC3>;
template class ImageView<Pixel16fC4>;
template class ImageView<Pixel16fC4A>;

ForAllChannelsConvertRoundWithAlpha(16f, 8u);
ForAllChannelsConvertRoundWithAlpha(16f, 8s);
ForAllChannelsConvertRoundWithAlpha(16f, 16u);
ForAllChannelsConvertRoundWithAlpha(16f, 16s);
ForAllChannelsConvertRoundWithAlpha(16f, 32u);
ForAllChannelsConvertRoundWithAlpha(16f, 32s);

InstantiateCopy_For(16f);

InstantiateSwapChannel_For(16f);

InstantiateDup_For(16f);

ForAllChannelsScaleAnyToIntWithAlpha(16f, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(16f, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(16f, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(16f, 64f);

} // namespace mpp::image::cpuSimple