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

template class ImageView<Pixel8sC1>;
template class ImageView<Pixel8sC2>;
template class ImageView<Pixel8sC3>;
template class ImageView<Pixel8sC4>;
template class ImageView<Pixel8sC4A>;

template class ImageView<Pixel16uC1>;
template class ImageView<Pixel16uC2>;
template class ImageView<Pixel16uC3>;
template class ImageView<Pixel16uC4>;
template class ImageView<Pixel16uC4A>;

template class ImageView<Pixel16sC1>;
template class ImageView<Pixel16sC2>;
template class ImageView<Pixel16sC3>;
template class ImageView<Pixel16sC4>;
template class ImageView<Pixel16sC4A>;

template class ImageView<Pixel16scC1>;
template class ImageView<Pixel16scC2>;
template class ImageView<Pixel16scC3>;
template class ImageView<Pixel16scC4>;

template class ImageView<Pixel32sC1>;
template class ImageView<Pixel32sC2>;
template class ImageView<Pixel32sC3>;
template class ImageView<Pixel32sC4>;
template class ImageView<Pixel32sC4A>;

template class ImageView<Pixel32uC1>;
template class ImageView<Pixel32uC2>;
template class ImageView<Pixel32uC3>;
template class ImageView<Pixel32uC4>;
template class ImageView<Pixel32uC4A>;

template class ImageView<Pixel32scC1>;
template class ImageView<Pixel32scC2>;
template class ImageView<Pixel32scC3>;
template class ImageView<Pixel32scC4>;

template class ImageView<Pixel16fC1>;
template class ImageView<Pixel16fC2>;
template class ImageView<Pixel16fC3>;
template class ImageView<Pixel16fC4>;
template class ImageView<Pixel16fC4A>;

template class ImageView<Pixel16bfC1>;
template class ImageView<Pixel16bfC2>;
template class ImageView<Pixel16bfC3>;
template class ImageView<Pixel16bfC4>;
template class ImageView<Pixel16bfC4A>;

template class ImageView<Pixel32fC1>;
template class ImageView<Pixel32fC2>;
template class ImageView<Pixel32fC3>;
template class ImageView<Pixel32fC4>;
template class ImageView<Pixel32fC4A>;

template class ImageView<Pixel32fcC1>;
template class ImageView<Pixel32fcC2>;
template class ImageView<Pixel32fcC3>;
template class ImageView<Pixel32fcC4>;

template class ImageView<Pixel64fC1>;
template class ImageView<Pixel64fC2>;
template class ImageView<Pixel64fC3>;
template class ImageView<Pixel64fC4>;
template class ImageView<Pixel64fC4A>;

ForAllChannelsConvertWithAlpha(8u, 8s);
ForAllChannelsConvertWithAlpha(8u, 16s);
ForAllChannelsConvertWithAlpha(8u, 16u);
ForAllChannelsConvertWithAlpha(8u, 32s);
ForAllChannelsConvertWithAlpha(8u, 32u);
ForAllChannelsConvertWithAlpha(8u, 16f);
ForAllChannelsConvertWithAlpha(8u, 16bf);
ForAllChannelsConvertWithAlpha(8u, 32f);
ForAllChannelsConvertWithAlpha(8u, 64f);

ForAllChannelsConvertWithAlpha(8s, 8u);
ForAllChannelsConvertWithAlpha(8s, 16u);
ForAllChannelsConvertWithAlpha(8s, 16s);
ForAllChannelsConvertWithAlpha(8s, 32u);
ForAllChannelsConvertWithAlpha(8s, 32s);
ForAllChannelsConvertWithAlpha(8s, 16f);
ForAllChannelsConvertWithAlpha(8s, 16bf);
ForAllChannelsConvertWithAlpha(8s, 32f);
ForAllChannelsConvertWithAlpha(8s, 64f);

ForAllChannelsConvertWithAlpha(16u, 8u);
ForAllChannelsConvertWithAlpha(16u, 8s);
ForAllChannelsConvertWithAlpha(16u, 16s);
ForAllChannelsConvertWithAlpha(16u, 32s);
ForAllChannelsConvertWithAlpha(16u, 32u);
ForAllChannelsConvertWithAlpha(16u, 16f);
ForAllChannelsConvertWithAlpha(16u, 16bf);
ForAllChannelsConvertWithAlpha(16u, 32f);
ForAllChannelsConvertWithAlpha(16u, 64f);

ForAllChannelsConvertWithAlpha(16s, 8u);
ForAllChannelsConvertWithAlpha(16s, 8s);
ForAllChannelsConvertWithAlpha(16s, 16u);
ForAllChannelsConvertWithAlpha(16s, 32s);
ForAllChannelsConvertWithAlpha(16s, 32u);
ForAllChannelsConvertWithAlpha(16s, 16f);
ForAllChannelsConvertWithAlpha(16s, 16bf);
ForAllChannelsConvertWithAlpha(16s, 32f);
ForAllChannelsConvertWithAlpha(16s, 64f);

ForAllChannelsConvertWithAlpha(32u, 8u);
ForAllChannelsConvertWithAlpha(32u, 8s);
ForAllChannelsConvertWithAlpha(32u, 16u);
ForAllChannelsConvertWithAlpha(32u, 16s);
ForAllChannelsConvertWithAlpha(32u, 32s);
ForAllChannelsConvertWithAlpha(32u, 16bf);
ForAllChannelsConvertWithAlpha(32u, 16f);
ForAllChannelsConvertWithAlpha(32u, 32f);
ForAllChannelsConvertWithAlpha(32u, 64f);

ForAllChannelsConvertWithAlpha(32s, 8u);
ForAllChannelsConvertWithAlpha(32s, 8s);
ForAllChannelsConvertWithAlpha(32s, 16u);
ForAllChannelsConvertWithAlpha(32s, 16s);
ForAllChannelsConvertWithAlpha(32s, 32u);
ForAllChannelsConvertWithAlpha(32s, 16bf);
ForAllChannelsConvertWithAlpha(32s, 16f);
ForAllChannelsConvertWithAlpha(32s, 32f);
ForAllChannelsConvertWithAlpha(32s, 64f);

ForAllChannelsConvertWithAlpha(32f, 16f);
ForAllChannelsConvertWithAlpha(32f, 16bf);
ForAllChannelsConvertWithAlpha(32f, 64f);

ForAllChannelsConvertWithAlpha(64f, 16f);
ForAllChannelsConvertWithAlpha(64f, 16bf);
ForAllChannelsConvertWithAlpha(64f, 32f);

ForAllChannelsConvertNoAlpha(16sc, 32sc);
ForAllChannelsConvertNoAlpha(16sc, 32fc);

ForAllChannelsConvertNoAlpha(32sc, 16sc);
ForAllChannelsConvertNoAlpha(32sc, 32fc);

ForAllChannelsConvertRoundWithAlpha(32f, 8u);
ForAllChannelsConvertRoundWithAlpha(32f, 8s);
ForAllChannelsConvertRoundWithAlpha(32f, 16u);
ForAllChannelsConvertRoundWithAlpha(32f, 16s);
ForAllChannelsConvertRoundWithAlpha(32f, 32u);
ForAllChannelsConvertRoundWithAlpha(32f, 32s);
ForAllChannelsConvertRoundWithAlpha(32f, 16bf);
ForAllChannelsConvertRoundWithAlpha(32f, 16f);

ForAllChannelsConvertRoundWithAlpha(16f, 8u);
ForAllChannelsConvertRoundWithAlpha(16f, 8s);
ForAllChannelsConvertRoundWithAlpha(16f, 16u);
ForAllChannelsConvertRoundWithAlpha(16f, 16s);
ForAllChannelsConvertRoundWithAlpha(16f, 32u);
ForAllChannelsConvertRoundWithAlpha(16f, 32s);

ForAllChannelsConvertRoundWithAlpha(16bf, 8u);
ForAllChannelsConvertRoundWithAlpha(16bf, 8s);
ForAllChannelsConvertRoundWithAlpha(16bf, 16u);
ForAllChannelsConvertRoundWithAlpha(16bf, 16s);
ForAllChannelsConvertRoundWithAlpha(16bf, 32u);
ForAllChannelsConvertRoundWithAlpha(16bf, 32s);

ForAllChannelsConvertRoundNoAlpha(32fc, 16sc);
ForAllChannelsConvertRoundNoAlpha(32fc, 32sc);

ForAllChannelsConvertRoundWithAlpha(64f, 8u);
ForAllChannelsConvertRoundWithAlpha(64f, 8s);
ForAllChannelsConvertRoundWithAlpha(64f, 16u);
ForAllChannelsConvertRoundWithAlpha(64f, 16s);
ForAllChannelsConvertRoundWithAlpha(64f, 32u);
ForAllChannelsConvertRoundWithAlpha(64f, 32s);

ForAllChannelsConvertRoundScaleWithAlpha(8u, 8s);

ForAllChannelsConvertRoundScaleWithAlpha(16u, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(16u, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(16u, 16s);

ForAllChannelsConvertRoundScaleWithAlpha(16s, 8s);

ForAllChannelsConvertRoundScaleWithAlpha(32u, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(32u, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(32u, 16s);
ForAllChannelsConvertRoundScaleWithAlpha(32u, 16u);
ForAllChannelsConvertRoundScaleWithAlpha(32u, 32s);

ForAllChannelsConvertRoundScaleWithAlpha(32s, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(32s, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(32s, 16s);
ForAllChannelsConvertRoundScaleWithAlpha(32s, 16u);

ForAllChannelsConvertRoundScaleWithAlpha(32f, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 16s);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 16u);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 32s);
ForAllChannelsConvertRoundScaleWithAlpha(32f, 32u);

ForAllChannelsConvertRoundScaleWithAlpha(64f, 8s);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 8u);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 16s);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 16u);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 32s);
ForAllChannelsConvertRoundScaleWithAlpha(64f, 32u);

ForAllChannelsConvertRoundScaleNoAlpha(32fc, 16sc);
ForAllChannelsConvertRoundScaleNoAlpha(32fc, 32sc);

ForAllChannelsConvertRoundScaleNoAlpha(32sc, 16sc);

InstantiateCopy_For(8u);
InstantiateCopy_For(8s);
InstantiateCopy_For(16u);
InstantiateCopy_For(16s);
InstantiateCopy_For(32u);
InstantiateCopy_For(32s);
InstantiateCopy_For(16f);
InstantiateCopy_For(16bf);
InstantiateCopy_For(32f);
InstantiateCopy_For(64f);
InstantiateCopy_For(16sc);
InstantiateCopy_For(32sc);
InstantiateCopy_For(32fc);

InstantiateSwapChannel_For(8u);
InstantiateSwapChannel_For(8s);
InstantiateSwapChannel_For(16u);
InstantiateSwapChannel_For(16s);
InstantiateSwapChannel_For(32u);
InstantiateSwapChannel_For(32s);
InstantiateSwapChannel_For(16f);
InstantiateSwapChannel_For(16bf);
InstantiateSwapChannel_For(32f);
InstantiateSwapChannel_For(64f);
InstantiateSwapChannel_For(16sc);
InstantiateSwapChannel_For(32sc);
InstantiateSwapChannel_For(32fc);

InstantiateDup_For(8u);
InstantiateDup_For(8s);
InstantiateDup_For(16u);
InstantiateDup_For(16s);
InstantiateDup_For(32u);
InstantiateDup_For(32s);
InstantiateDup_For(16f);
InstantiateDup_For(16bf);
InstantiateDup_For(32f);
InstantiateDup_For(64f);
InstantiateDupNoAlpha_For(16sc);
InstantiateDupNoAlpha_For(32sc);
InstantiateDupNoAlpha_For(32fc);

ForAllChannelsScaleIntToIntWithAlpha(8s, 8u);
ForAllChannelsScaleIntToIntWithAlpha(8s, 16s);
ForAllChannelsScaleIntToIntWithAlpha(8s, 16u);
ForAllChannelsScaleIntToIntWithAlpha(8s, 32s);
ForAllChannelsScaleIntToIntWithAlpha(8s, 32u);

ForAllChannelsScaleIntToIntWithAlpha(8u, 8s);
ForAllChannelsScaleIntToIntWithAlpha(8u, 16s);
ForAllChannelsScaleIntToIntWithAlpha(8u, 16u);
ForAllChannelsScaleIntToIntWithAlpha(8u, 32s);
ForAllChannelsScaleIntToIntWithAlpha(8u, 32u);

ForAllChannelsScaleIntToIntWithAlpha(16s, 8u);
ForAllChannelsScaleIntToIntWithAlpha(16s, 8s);
ForAllChannelsScaleIntToIntWithAlpha(16s, 16u);
ForAllChannelsScaleIntToIntWithAlpha(16s, 32s);
ForAllChannelsScaleIntToIntWithAlpha(16s, 32u);

ForAllChannelsScaleIntToIntWithAlpha(16u, 8s);
ForAllChannelsScaleIntToIntWithAlpha(16u, 8u);
ForAllChannelsScaleIntToIntWithAlpha(16u, 16s);
ForAllChannelsScaleIntToIntWithAlpha(16u, 32s);
ForAllChannelsScaleIntToIntWithAlpha(16u, 32u);

ForAllChannelsScaleIntToIntWithAlpha(32s, 8u);
ForAllChannelsScaleIntToIntWithAlpha(32s, 8s);
ForAllChannelsScaleIntToIntWithAlpha(32s, 16u);
ForAllChannelsScaleIntToIntWithAlpha(32s, 16s);
ForAllChannelsScaleIntToIntWithAlpha(32s, 32u);

ForAllChannelsScaleIntToIntWithAlpha(32u, 8s);
ForAllChannelsScaleIntToIntWithAlpha(32u, 8u);
ForAllChannelsScaleIntToIntWithAlpha(32u, 16s);
ForAllChannelsScaleIntToIntWithAlpha(32u, 16u);
ForAllChannelsScaleIntToIntWithAlpha(32u, 32s);

ForAllChannelsScaleIntToIntNoAlpha(16sc, 32sc);
ForAllChannelsScaleIntToIntNoAlpha(32sc, 16sc);

ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 32s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8s, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(8s, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(8s, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(8s, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(8s, 64f);

ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 32s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(8u, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(8u, 64f);

ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 32s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16s, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(16s, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(16s, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(16s, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(16s, 64f);

ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 32s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(16u, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(16u, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(16u, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(16u, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(16u, 64f);

ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32s, 32u);
ForAllChannelsScaleIntToAnyWithAlpha(32s, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(32s, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(32s, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(32s, 64f);

ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 8s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 8u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 16s);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 16u);
ForAllChannelsScaleIntToAnyRoundWithAlpha(32u, 32s);
ForAllChannelsScaleIntToAnyWithAlpha(32u, 16f);
ForAllChannelsScaleIntToAnyWithAlpha(32u, 16bf);
ForAllChannelsScaleIntToAnyWithAlpha(32u, 32f);
ForAllChannelsScaleIntToAnyWithAlpha(32u, 64f);

ForAllChannelsScaleIntToAnyRoundNoAlpha(16sc, 32sc);
ForAllChannelsScaleIntToAnyNoAlpha(16sc, 32fc);
ForAllChannelsScaleIntToAnyRoundNoAlpha(32sc, 16sc);
ForAllChannelsScaleIntToAnyNoAlpha(32sc, 32fc);

ForAllChannelsScaleAnyToIntWithAlpha(8s, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(8s, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(8s, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(8s, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(8s, 32u);

ForAllChannelsScaleAnyToIntWithAlpha(8u, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(8u, 32u);

ForAllChannelsScaleAnyToIntWithAlpha(16s, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(16s, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(16s, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(16s, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(16s, 32u);

ForAllChannelsScaleAnyToIntWithAlpha(16u, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(16u, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(16u, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(16u, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(16u, 32u);

ForAllChannelsScaleAnyToIntWithAlpha(32s, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(32s, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(32s, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(32s, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(32s, 32u);

ForAllChannelsScaleAnyToIntWithAlpha(32u, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(32u, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(32u, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(32u, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(32u, 32s);

ForAllChannelsScaleAnyToIntWithAlpha(16f, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(16f, 32u);

ForAllChannelsScaleAnyToIntWithAlpha(16bf, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(16bf, 32u);

ForAllChannelsScaleAnyToIntWithAlpha(32f, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(32f, 32u);

ForAllChannelsScaleAnyToIntWithAlpha(64f, 8s);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 8u);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 16s);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 16u);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 32s);
ForAllChannelsScaleAnyToIntWithAlpha(64f, 32u);

ForAllChannelsScaleAnyToIntNoAlpha(16sc, 32sc);
ForAllChannelsScaleAnyToIntNoAlpha(32sc, 16sc);
ForAllChannelsScaleAnyToIntNoAlpha(32fc, 16sc);
ForAllChannelsScaleAnyToIntNoAlpha(32fc, 32sc);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8s, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(8s, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(8s, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(8s, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(8s, 64f);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(8u, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(8u, 64f);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16s, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(16s, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(16s, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(16s, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(16s, 64f);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16u, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(16u, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(16u, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(16u, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(16u, 64f);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32s, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(32s, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(32s, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(32s, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(32s, 64f);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32u, 32s);
ForAllChannelsScaleAnyToAnyWithAlpha(32u, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(32u, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(32u, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(32u, 64f);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16f, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(16f, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(16f, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(16f, 64f);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(16bf, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(16bf, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(16bf, 32f);
ForAllChannelsScaleAnyToAnyWithAlpha(16bf, 64f);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(32f, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(32f, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(32f, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(32f, 64f);

ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 8s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 8u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 16s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 16u);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 32s);
ForAllChannelsScaleAnyToAnyRoundWithAlpha(64f, 32u);
ForAllChannelsScaleAnyToAnyWithAlpha(64f, 16f);
ForAllChannelsScaleAnyToAnyWithAlpha(64f, 16bf);
ForAllChannelsScaleAnyToAnyWithAlpha(64f, 32f);

ForAllChannelsScaleAnyToAnyRoundNoAlpha(16sc, 32sc);
ForAllChannelsScaleAnyToAnyNoAlpha(16sc, 32fc);
ForAllChannelsScaleAnyToAnyRoundNoAlpha(32sc, 16sc);
ForAllChannelsScaleAnyToAnyNoAlpha(32sc, 32fc);
ForAllChannelsScaleAnyToAnyRoundNoAlpha(32fc, 16sc);
ForAllChannelsScaleAnyToAnyRoundNoAlpha(32fc, 32sc);

} // namespace mpp::image::cpuSimple