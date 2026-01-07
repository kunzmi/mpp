#include "../dllexport_simplecpu.h" //NOLINT
#define MPPEXPORT_SIMPLECPU8S MPPEXPORT_SIMPLECPU
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
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel8sC1>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel8sC2>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel8sC3>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel8sC4>;
extern template class MPPEXPORT_SIMPLECPU ImageView<Pixel8sC4A>;
#endif

template class ImageView<Pixel8sC1>;
template class ImageView<Pixel8sC2>;
template class ImageView<Pixel8sC3>;
template class ImageView<Pixel8sC4>;
template class ImageView<Pixel8sC4A>;

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

} // namespace mpp::image::cpuSimple