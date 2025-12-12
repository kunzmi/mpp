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
template class ImageView<Pixel8uC1>;
template class ImageView<Pixel8uC2>;
template class ImageView<Pixel8uC3>;
template class ImageView<Pixel8uC4>;
template class ImageView<Pixel8uC4A>;

ForAllChannelsConvertWithAlpha(8u, 8s);
ForAllChannelsConvertWithAlpha(8u, 16s);
ForAllChannelsConvertWithAlpha(8u, 16u);
ForAllChannelsConvertWithAlpha(8u, 32s);
ForAllChannelsConvertWithAlpha(8u, 32u);
ForAllChannelsConvertWithAlpha(8u, 16f);
ForAllChannelsConvertWithAlpha(8u, 16bf);
ForAllChannelsConvertWithAlpha(8u, 32f);
ForAllChannelsConvertWithAlpha(8u, 64f);

ForAllChannelsConvertRoundScaleWithAlpha(8u, 8s);

InstantiateCopy_For(8u);

InstantiateSwapChannel_For(8u);

InstantiateDup_For(8u);

ForAllChannelsScaleIntToIntWithAlpha(8u, 8s);
ForAllChannelsScaleIntToIntWithAlpha(8u, 16s);
ForAllChannelsScaleIntToIntWithAlpha(8u, 16u);
ForAllChannelsScaleIntToIntWithAlpha(8u, 32s);
ForAllChannelsScaleIntToIntWithAlpha(8u, 32u);

ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 32s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 64f);

ForAllChannelsScaleAnyToIntWithAlpha(8u, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 32u);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 64f);

} // namespace mpp::image::cpuSimple