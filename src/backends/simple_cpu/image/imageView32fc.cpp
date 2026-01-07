#include "../dllexport_simplecpu.h" //NOLINT
#define MPPEXPORT_SIMPLECPU32FC MPPEXPORT_SIMPLECPU
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
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32fcC1>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32fcC2>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32fcC3>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel32fcC4>;
#endif

template class ImageView<Pixel32fcC1>;
template class ImageView<Pixel32fcC2>;
template class ImageView<Pixel32fcC3>;
template class ImageView<Pixel32fcC4>;

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

} // namespace mpp::image::cpuSimple