#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND
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
template class ImageView<Pixel16scC1>;
template class ImageView<Pixel16scC2>;
template class ImageView<Pixel16scC3>;
template class ImageView<Pixel16scC4>;
template <> ImageView<Pixel16scC1> ImageView<Pixel16scC1>::Null = ImageView<Pixel16scC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16scC2> ImageView<Pixel16scC2>::Null = ImageView<Pixel16scC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16scC3> ImageView<Pixel16scC3>::Null = ImageView<Pixel16scC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16scC4> ImageView<Pixel16scC4>::Null = ImageView<Pixel16scC4>(nullptr, Size2D(0, 0), 0);

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

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND