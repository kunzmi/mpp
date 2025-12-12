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
#include <common/complex.h>                                 //NOLINT(misc-include-cleaner)
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h> //NOLINT(misc-include-cleaner)

namespace mpp::image::cpuSimple
{
template class ImageView<Pixel16scC1>;
template class ImageView<Pixel16scC2>;
template class ImageView<Pixel16scC3>;
template class ImageView<Pixel16scC4>;

ForAllChannelsConvertNoAlpha(16sc, 32sc);
ForAllChannelsConvertNoAlpha(16sc, 32fc);

InstantiateCopy_For(16sc);

InstantiateSwapChannel_For(16sc);

InstantiateDupNoAlpha_For(16sc);

ForAllChannelsScaleIntToIntNoAlpha(16sc, 32sc);

ForAllChannelsScaleIntToAnyRoundNoAlpha(16sc, 32sc);
ForAllChannelsScaleIntToAnyNoAlpha(16sc, 32fc);

ForAllChannelsScaleAnyToIntNoAlpha(16sc, 32sc);

ForAllChannelsScaleAnyToAnyRoundNoAlpha(16sc, 32sc);
ForAllChannelsScaleAnyToAnyNoAlpha(16sc, 32fc);

} // namespace mpp::image::cpuSimple