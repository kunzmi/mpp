#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include "imageView.h"
#include "imageView_arithmetic_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_dataExchangeAndInit_impl.h" //NOLINT(misc-include-cleaner)
#include "imageView_filtering_impl.h"           //NOLINT(misc-include-cleaner)
#include "imageView_morphology_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_geometryTransforms_impl.h"  //NOLINT(misc-include-cleaner)
#include "imageView_statistics_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_thresholdAndCompare_impl.h" //NOLINT(misc-include-cleaner)
#include <backends/cuda/streamCtx.h>            //NOLINT(misc-include-cleaner)
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h> //NOLINT(misc-include-cleaner)
#include "dataExchangeAndInit/instantiateConversion.h"

namespace mpp::image::cuda
{
template class ImageView<Pixel8uC1>;
template class ImageView<Pixel8uC2>;
template class ImageView<Pixel8uC3>;
template class ImageView<Pixel8uC4>;
template class ImageView<Pixel8uC4A>;

template <> ImageView<Pixel8uC1> ImageView<Pixel8uC1>::Null = ImageView<Pixel8uC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8uC2> ImageView<Pixel8uC2>::Null = ImageView<Pixel8uC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8uC3> ImageView<Pixel8uC3>::Null = ImageView<Pixel8uC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8uC4> ImageView<Pixel8uC4>::Null = ImageView<Pixel8uC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8uC4A> ImageView<Pixel8uC4A>::Null = ImageView<Pixel8uC4A>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel8sC1>;
template class ImageView<Pixel8sC2>;
template class ImageView<Pixel8sC3>;
template class ImageView<Pixel8sC4>;
template class ImageView<Pixel8sC4A>;

template <> ImageView<Pixel8sC1> ImageView<Pixel8sC1>::Null   = ImageView<Pixel8sC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8sC2> ImageView<Pixel8sC2>::Null   = ImageView<Pixel8sC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8sC3> ImageView<Pixel8sC3>::Null   = ImageView<Pixel8sC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8sC4> ImageView<Pixel8sC4>::Null   = ImageView<Pixel8sC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel8sC4A> ImageView<Pixel8sC4A>::Null = ImageView<Pixel8sC4A>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel16uC1>;
template class ImageView<Pixel16uC2>;
template class ImageView<Pixel16uC3>;
template class ImageView<Pixel16uC4>;
template class ImageView<Pixel16uC4A>;

template <> ImageView<Pixel16uC1> ImageView<Pixel16uC1>::Null = ImageView<Pixel16uC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16uC2> ImageView<Pixel16uC2>::Null = ImageView<Pixel16uC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16uC3> ImageView<Pixel16uC3>::Null = ImageView<Pixel16uC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16uC4> ImageView<Pixel16uC4>::Null = ImageView<Pixel16uC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16uC4A> ImageView<Pixel16uC4A>::Null = ImageView<Pixel16uC4A>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel16sC1>;
template class ImageView<Pixel16sC2>;
template class ImageView<Pixel16sC3>;
template class ImageView<Pixel16sC4>;
template class ImageView<Pixel16sC4A>;
template <> ImageView<Pixel16sC1> ImageView<Pixel16sC1>::Null = ImageView<Pixel16sC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16sC2> ImageView<Pixel16sC2>::Null = ImageView<Pixel16sC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16sC3> ImageView<Pixel16sC3>::Null = ImageView<Pixel16sC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16sC4> ImageView<Pixel16sC4>::Null = ImageView<Pixel16sC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16sC4A> ImageView<Pixel16sC4A>::Null = ImageView<Pixel16sC4A>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel16scC1>;
template class ImageView<Pixel16scC2>;
template class ImageView<Pixel16scC3>;
template class ImageView<Pixel16scC4>;
template <> ImageView<Pixel16scC1> ImageView<Pixel16scC1>::Null = ImageView<Pixel16scC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16scC2> ImageView<Pixel16scC2>::Null = ImageView<Pixel16scC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16scC3> ImageView<Pixel16scC3>::Null = ImageView<Pixel16scC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16scC4> ImageView<Pixel16scC4>::Null = ImageView<Pixel16scC4>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel32sC1>;
template class ImageView<Pixel32sC2>;
template class ImageView<Pixel32sC3>;
template class ImageView<Pixel32sC4>;
template class ImageView<Pixel32sC4A>;
template <> ImageView<Pixel32sC1> ImageView<Pixel32sC1>::Null = ImageView<Pixel32sC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32sC2> ImageView<Pixel32sC2>::Null = ImageView<Pixel32sC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32sC3> ImageView<Pixel32sC3>::Null = ImageView<Pixel32sC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32sC4> ImageView<Pixel32sC4>::Null = ImageView<Pixel32sC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32sC4A> ImageView<Pixel32sC4A>::Null = ImageView<Pixel32sC4A>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel32uC1>;
template class ImageView<Pixel32uC2>;
template class ImageView<Pixel32uC3>;
template class ImageView<Pixel32uC4>;
template class ImageView<Pixel32uC4A>;
template <> ImageView<Pixel32uC1> ImageView<Pixel32uC1>::Null = ImageView<Pixel32uC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32uC2> ImageView<Pixel32uC2>::Null = ImageView<Pixel32uC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32uC3> ImageView<Pixel32uC3>::Null = ImageView<Pixel32uC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32uC4> ImageView<Pixel32uC4>::Null = ImageView<Pixel32uC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32uC4A> ImageView<Pixel32uC4A>::Null = ImageView<Pixel32uC4A>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel32scC1>;
template class ImageView<Pixel32scC2>;
template class ImageView<Pixel32scC3>;
template class ImageView<Pixel32scC4>;
template <> ImageView<Pixel32scC1> ImageView<Pixel32scC1>::Null = ImageView<Pixel32scC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32scC2> ImageView<Pixel32scC2>::Null = ImageView<Pixel32scC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32scC3> ImageView<Pixel32scC3>::Null = ImageView<Pixel32scC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32scC4> ImageView<Pixel32scC4>::Null = ImageView<Pixel32scC4>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel16fC1>;
template class ImageView<Pixel16fC2>;
template class ImageView<Pixel16fC3>;
template class ImageView<Pixel16fC4>;
template class ImageView<Pixel16fC4A>;
template <> ImageView<Pixel16fC1> ImageView<Pixel16fC1>::Null   = ImageView<Pixel16fC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16fC2> ImageView<Pixel16fC2>::Null   = ImageView<Pixel16fC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16fC3> ImageView<Pixel16fC3>::Null   = ImageView<Pixel16fC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16fC4> ImageView<Pixel16fC4>::Null   = ImageView<Pixel16fC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16fC4A> ImageView<Pixel16fC4A>::Null = ImageView<Pixel16fC4A>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel16bfC1>;
template class ImageView<Pixel16bfC2>;
template class ImageView<Pixel16bfC3>;
template class ImageView<Pixel16bfC4>;
template class ImageView<Pixel16bfC4A>;
template <> ImageView<Pixel16bfC1> ImageView<Pixel16bfC1>::Null   = ImageView<Pixel16bfC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16bfC2> ImageView<Pixel16bfC2>::Null   = ImageView<Pixel16bfC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16bfC3> ImageView<Pixel16bfC3>::Null   = ImageView<Pixel16bfC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16bfC4> ImageView<Pixel16bfC4>::Null   = ImageView<Pixel16bfC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel16bfC4A> ImageView<Pixel16bfC4A>::Null = ImageView<Pixel16bfC4A>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel32fC1>;
template class ImageView<Pixel32fC2>;
template class ImageView<Pixel32fC3>;
template class ImageView<Pixel32fC4>;
template class ImageView<Pixel32fC4A>;
template <> ImageView<Pixel32fC1> ImageView<Pixel32fC1>::Null   = ImageView<Pixel32fC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC2> ImageView<Pixel32fC2>::Null   = ImageView<Pixel32fC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC3> ImageView<Pixel32fC3>::Null   = ImageView<Pixel32fC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC4> ImageView<Pixel32fC4>::Null   = ImageView<Pixel32fC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC4A> ImageView<Pixel32fC4A>::Null = ImageView<Pixel32fC4A>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel32fcC1>;
template class ImageView<Pixel32fcC2>;
template class ImageView<Pixel32fcC3>;
template class ImageView<Pixel32fcC4>;
template <> ImageView<Pixel32fcC1> ImageView<Pixel32fcC1>::Null = ImageView<Pixel32fcC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fcC2> ImageView<Pixel32fcC2>::Null = ImageView<Pixel32fcC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fcC3> ImageView<Pixel32fcC3>::Null = ImageView<Pixel32fcC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fcC4> ImageView<Pixel32fcC4>::Null = ImageView<Pixel32fcC4>(nullptr, Size2D(0, 0), 0);

template class ImageView<Pixel64fC1>;
template class ImageView<Pixel64fC2>;
template class ImageView<Pixel64fC3>;
template class ImageView<Pixel64fC4>;
template class ImageView<Pixel64fC4A>;
template <> ImageView<Pixel64fC1> ImageView<Pixel64fC1>::Null   = ImageView<Pixel64fC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel64fC2> ImageView<Pixel64fC2>::Null   = ImageView<Pixel64fC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel64fC3> ImageView<Pixel64fC3>::Null   = ImageView<Pixel64fC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel64fC4> ImageView<Pixel64fC4>::Null   = ImageView<Pixel64fC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel64fC4A> ImageView<Pixel64fC4A>::Null = ImageView<Pixel64fC4A>(nullptr, Size2D(0, 0), 0);



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
ForAllChannelsConvertWithAlpha(16u, 32s);
ForAllChannelsConvertWithAlpha(16u, 32u);
ForAllChannelsConvertWithAlpha(16u, 16f);
ForAllChannelsConvertWithAlpha(16u, 16bf);
ForAllChannelsConvertWithAlpha(16u, 32f);
ForAllChannelsConvertWithAlpha(16u, 64f);

ForAllChannelsConvertWithAlpha(16s, 8u);
ForAllChannelsConvertWithAlpha(16s, 16u);
ForAllChannelsConvertWithAlpha(16s, 32s);
ForAllChannelsConvertWithAlpha(16s, 32u);
ForAllChannelsConvertWithAlpha(16s, 16f);
ForAllChannelsConvertWithAlpha(16s, 16bf);
ForAllChannelsConvertWithAlpha(16s, 32f);
ForAllChannelsConvertWithAlpha(16s, 64f);

ForAllChannelsConvertWithAlpha(32u, 8u);
ForAllChannelsConvertWithAlpha(32u, 16u);
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
ForAllChannelsConvertNoAlpha(16sc, 64fc);

ForAllChannelsConvertNoAlpha(32sc, 32fc);
ForAllChannelsConvertNoAlpha(32sc, 64fc);

ForAllChannelsConvertRoundWithAlpha(32f, 8u);
ForAllChannelsConvertRoundWithAlpha(32f, 8s);
ForAllChannelsConvertRoundWithAlpha(32f, 16u);
ForAllChannelsConvertRoundWithAlpha(32f, 16s);
ForAllChannelsConvertRoundWithAlpha(32f, 16bf);
ForAllChannelsConvertRoundWithAlpha(32f, 16f);

ForAllChannelsConvertRoundWithAlpha(16f, 8u);
ForAllChannelsConvertRoundWithAlpha(16f, 8s);
ForAllChannelsConvertRoundWithAlpha(16f, 16u);
ForAllChannelsConvertRoundWithAlpha(16f, 16s);

ForAllChannelsConvertRoundWithAlpha(16bf, 8u);
ForAllChannelsConvertRoundWithAlpha(16bf, 8s);
ForAllChannelsConvertRoundWithAlpha(16bf, 16u);
ForAllChannelsConvertRoundWithAlpha(16bf, 16s);

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

//template ImageView<Pixel32fC1> &ImageView<Pixel8uC1>::Convert<Pixel32fC1>(ImageView<Pixel32fC1> &aDst,
//                                                                          const mpp::cuda::StreamCtx &aStreamCtx) const;
//template ImageView<Pixel32fC3> &ImageView<Pixel8uC3>::Convert<Pixel32fC3>(ImageView<Pixel32fC3> &aDst,
//                                                                          const mpp::cuda::StreamCtx &aStreamCtx) const;
//template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
//                                                                         RoundingMode aRoundingMode,
//                                                                         const mpp::cuda::StreamCtx &aStreamCtx) const;
//template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
//                                                                         RoundingMode aRoundingMode, int aScaleFactor,
//                                                                         const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC2>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC2>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC2>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC3>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC3>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC3>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC4>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC4>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC4>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC1>::Copy<Pixel8uC2>(ImageView<Pixel8uC2> &aDst, Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC1>::Copy<Pixel8uC3>(ImageView<Pixel8uC3> &aDst, Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC1>::Copy<Pixel8uC4>(ImageView<Pixel8uC4> &aDst, Channel aDstChannel,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC1> &ImageView<Pixel8uC2>::Copy<Pixel8uC1>(Channel aSrcChannel, ImageView<Pixel8uC1> &aDst,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC1> &ImageView<Pixel8uC3>::Copy<Pixel8uC1>(Channel aSrcChannel, ImageView<Pixel8uC1> &aDst,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC1> &ImageView<Pixel8uC4>::Copy<Pixel8uC1>(Channel aSrcChannel, ImageView<Pixel8uC1> &aDst,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC3> &ImageView<Pixel8uC1>::Dup<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                    const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC1>::Dup<Pixel8uC4>(ImageView<Pixel8uC4> &aDst,
                                                                    const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4A> &ImageView<Pixel8uC1>::Dup<Pixel8uC4A>(ImageView<Pixel8uC4A> &aDst,
                                                                      const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC4> &ImageView<Pixel8uC3>::SwapChannel<Pixel8uC4>(
    ImageView<Pixel8uC4> &aDst, const ChannelList<vector_active_size_v<Pixel8uC4>> &aDstChannels,
    remove_vector_t<Pixel8uC3> aValue, const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC4>::SwapChannel<Pixel8uC3>(
    ImageView<Pixel8uC3> &aDst, const ChannelList<vector_active_size_v<Pixel8uC3>> &aDstChannels,
    const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC3>::SwapChannel<Pixel8uC3>(
    ImageView<Pixel8uC3> &aDst, const ChannelList<vector_active_size_v<Pixel8uC3>> &aDstChannels,
    const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC4>::SwapChannel<Pixel8uC4>(
    ImageView<Pixel8uC4> &aDst, const ChannelList<vector_active_size_v<Pixel8uC4>> &aDstChannels,
    const mpp::cuda::StreamCtx &aStreamCtx) const;


//template ImageView<Pixel32fC3> &ImageView<Pixel16uC3>::Convert<Pixel32fC3>(
//    ImageView<Pixel32fC3> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;
//template ImageView<Pixel16uC3> &ImageView<Pixel32fC3>::Convert<Pixel16uC3>(
//    ImageView<Pixel16uC3> &aDst, RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx) const;
//template ImageView<Pixel16uC3> &ImageView<Pixel32fC3>::Convert<Pixel16uC3>(
//    ImageView<Pixel16uC3> &aDst, RoundingMode aRoundingMode, int aScaleFactor,
//    const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC2> &ImageView<Pixel16uC2>::Copy<Pixel16uC2>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC2> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC2>::Copy<Pixel16uC3>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC3> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC2>::Copy<Pixel16uC4>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC4> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC2> &ImageView<Pixel16uC3>::Copy<Pixel16uC2>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC2> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC3>::Copy<Pixel16uC3>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC3> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC3>::Copy<Pixel16uC4>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC4> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC2> &ImageView<Pixel16uC4>::Copy<Pixel16uC2>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC2> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC4>::Copy<Pixel16uC3>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC3> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC4>::Copy<Pixel16uC4>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC4> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC2> &ImageView<Pixel16uC1>::Copy<Pixel16uC2>(ImageView<Pixel16uC2> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC1>::Copy<Pixel16uC3>(ImageView<Pixel16uC3> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC1>::Copy<Pixel16uC4>(ImageView<Pixel16uC4> &aDst,
                                                                        Channel aDstChannel,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC1> &ImageView<Pixel16uC2>::Copy<Pixel16uC1>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC1> &aDst,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC1> &ImageView<Pixel16uC3>::Copy<Pixel16uC1>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC1> &aDst,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC1> &ImageView<Pixel16uC4>::Copy<Pixel16uC1>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC1> &aDst,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC3> &ImageView<Pixel16uC1>::Dup<Pixel16uC3>(ImageView<Pixel16uC3> &aDst,
                                                                       const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC1>::Dup<Pixel16uC4>(ImageView<Pixel16uC4> &aDst,
                                                                       const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4A> &ImageView<Pixel16uC1>::Dup<Pixel16uC4A>(ImageView<Pixel16uC4A> &aDst,
                                                                         const mpp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC4> &ImageView<Pixel16uC3>::SwapChannel<Pixel16uC4>(
    ImageView<Pixel16uC4> &aDst, const ChannelList<vector_active_size_v<Pixel16uC4>> &aDstChannels,
    remove_vector_t<Pixel16uC3> aValue, const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC4>::SwapChannel<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, const ChannelList<vector_active_size_v<Pixel16uC3>> &aDstChannels,
    const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC3>::SwapChannel<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, const ChannelList<vector_active_size_v<Pixel16uC3>> &aDstChannels,
    const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC4>::SwapChannel<Pixel16uC4>(
    ImageView<Pixel16uC4> &aDst, const ChannelList<vector_active_size_v<Pixel16uC4>> &aDstChannels,
    const mpp::cuda::StreamCtx &aStreamCtx) const;




//#pragma region Instantiate Affine
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateAffine_For(pixelT)                                                                                  \
//    template ImageView<pixelT> &ImageView<pixelT>::WarpAffine(                                                         \
//        ImageView<pixelT> &aDst, const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,        \
//        BorderType aBorder, Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const;                        \
//    template ImageView<pixelT> &ImageView<pixelT>::WarpAffine(                                                         \
//        ImageView<pixelT> &aDst, const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,        \
//        BorderType aBorder, pixelT aConstant, Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const;      \
//    template ImageView<pixelT> &ImageView<pixelT>::WarpAffineBack(                                                     \
//        ImageView<pixelT> &aDst, const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,        \
//        BorderType aBorder, Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const;                        \
//    template ImageView<pixelT> &ImageView<pixelT>::WarpAffineBack(                                                     \
//        ImageView<pixelT> &aDst, const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,        \
//        BorderType aBorder, pixelT aConstant, Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const;
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateAffinePlanar_For(pixelT)                                                                            \
//    template void ImageView<pixelT##C4>::WarpAffine(ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4,            \
//                                                    const AffineTransformation<double> &aAffine,                       \
//                                                    InterpolationMode aInterpolation, BorderType aBorder,              \
//                                                    Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);      \
//    template void ImageView<pixelT##C4>::WarpAffine(                                                                   \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4, const AffineTransformation<double> &aAffine,           \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C4 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//    template void ImageView<pixelT##C4>::WarpAffineBack(ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4,        \
//                                                        const AffineTransformation<double> &aAffine,                   \
//                                                        InterpolationMode aInterpolation, BorderType aBorder,          \
//                                                        Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);  \
//    template void ImageView<pixelT##C4>::WarpAffineBack(                                                               \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4, const AffineTransformation<double> &aAffine,           \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C4 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//                                                                                                                       \
//    template void ImageView<pixelT##C3>::WarpAffine(ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc1,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc2,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst1,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst2,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3,            \
//                                                    const AffineTransformation<double> &aAffine,                       \
//                                                    InterpolationMode aInterpolation, BorderType aBorder,              \
//                                                    Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);      \
//    template void ImageView<pixelT##C3>::WarpAffine(                                                                   \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3, const AffineTransformation<double> &aAffine,           \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C3 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//    template void ImageView<pixelT##C3>::WarpAffineBack(ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc1,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc2,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst1,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst2,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3,        \
//                                                        const AffineTransformation<double> &aAffine,                   \
//                                                        InterpolationMode aInterpolation, BorderType aBorder,          \
//                                                        Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);  \
//    template void ImageView<pixelT##C3>::WarpAffineBack(                                                               \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3, const AffineTransformation<double> &aAffine,           \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C3 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//                                                                                                                       \
//    template void ImageView<pixelT##C2>::WarpAffine(ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,            \
//                                                    ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2,            \
//                                                    const AffineTransformation<double> &aAffine,                       \
//                                                    InterpolationMode aInterpolation, BorderType aBorder,              \
//                                                    Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);      \
//    template void ImageView<pixelT##C2>::WarpAffine(                                                                   \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2, const AffineTransformation<double> &aAffine,           \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C2 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//    template void ImageView<pixelT##C2>::WarpAffineBack(ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,        \
//                                                        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2,        \
//                                                        const AffineTransformation<double> &aAffine,                   \
//                                                        InterpolationMode aInterpolation, BorderType aBorder,          \
//                                                        Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);  \
//    template void ImageView<pixelT##C2>::WarpAffineBack(                                                               \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2, const AffineTransformation<double> &aAffine,           \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C2 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateGeomTransform_For(pixelT) InstantiateAffine_For(pixelT);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define ForAllChannelsNoAlpha(type)                                                                                    \
//    InstantiateGeomTransform_For(Pixel##type##C1);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C2);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C3);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4);                                                                     \
//    InstantiateAffinePlanar_For(Pixel##type);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define ForAllChannelsWithAlpha(type)                                                                                  \
//    InstantiateGeomTransform_For(Pixel##type##C1);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C2);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C3);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4A);                                                                    \
//    InstantiateAffinePlanar_For(Pixel##type);
//
//ForAllChannelsWithAlpha(8u);
//// ForAllChannelsWithAlpha(8s);
////
//// ForAllChannelsWithAlpha(16u);
//// ForAllChannelsWithAlpha(16s);
////
//// ForAllChannelsWithAlpha(32u);
//// ForAllChannelsWithAlpha(32s);
////
//// ForAllChannelsWithAlpha(16f);
//// ForAllChannelsWithAlpha(16bf);
//// ForAllChannelsWithAlpha(32f);
//// ForAllChannelsWithAlpha(64f);
////
//// ForAllChannelsNoAlpha(16sc);
//// ForAllChannelsNoAlpha(32sc);
//// ForAllChannelsNoAlpha(32fc);
//
//#undef InstantiateAffinePlanar_For
//#undef InstantiateAffine_For
//#undef InstantiateGeomTransform_For
//#undef ForAllChannelsWithAlpha
//#undef ForAllChannelsNoAlpha
//#pragma endregion
//
//#pragma region Instantiate Perspective
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiatePerspective_For(pixelT)                                                                             \
//    template ImageView<pixelT> &ImageView<pixelT>::WarpPerspective(                                                    \
//        ImageView<pixelT> &aDst, const PerspectiveTransformation<double> &aPerspective,                                \
//        InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi,                                     \
//        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
//    template ImageView<pixelT> &ImageView<pixelT>::WarpPerspective(                                                    \
//        ImageView<pixelT> &aDst, const PerspectiveTransformation<double> &aPerspective,                                \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT aConstant, Roi aAllowedReadRoi,                   \
//        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
//    template ImageView<pixelT> &ImageView<pixelT>::WarpPerspectiveBack(                                                \
//        ImageView<pixelT> &aDst, const PerspectiveTransformation<double> &aPerspective,                                \
//        InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi,                                     \
//        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
//    template ImageView<pixelT> &ImageView<pixelT>::WarpPerspectiveBack(                                                \
//        ImageView<pixelT> &aDst, const PerspectiveTransformation<double> &aPerspective,                                \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT aConstant, Roi aAllowedReadRoi,                   \
//        const mpp::cuda::StreamCtx &aStreamCtx) const;
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiatePerspectivePlanar_For(pixelT)                                                                       \
//    template void ImageView<pixelT##C4>::WarpPerspective(ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4,       \
//                                                         const PerspectiveTransformation<double> &aPerspective,        \
//                                                         InterpolationMode aInterpolation, BorderType aBorder,         \
//                                                         Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx); \
//    template void ImageView<pixelT##C4>::WarpPerspective(                                                              \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4, const PerspectiveTransformation<double> &aPerspective, \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C4 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//    template void ImageView<pixelT##C4>::WarpPerspectiveBack(                                                          \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4, const PerspectiveTransformation<double> &aPerspective, \
//        InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi,                                     \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//    template void ImageView<pixelT##C4>::WarpPerspectiveBack(                                                          \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4, const PerspectiveTransformation<double> &aPerspective, \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C4 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//                                                                                                                       \
//    template void ImageView<pixelT##C3>::WarpPerspective(ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc1,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc2,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst1,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst2,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3,       \
//                                                         const PerspectiveTransformation<double> &aPerspective,        \
//                                                         InterpolationMode aInterpolation, BorderType aBorder,         \
//                                                         Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx); \
//    template void ImageView<pixelT##C3>::WarpPerspective(                                                              \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3, const PerspectiveTransformation<double> &aPerspective, \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C3 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//    template void ImageView<pixelT##C3>::WarpPerspectiveBack(                                                          \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3, const PerspectiveTransformation<double> &aPerspective, \
//        InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi,                                     \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//    template void ImageView<pixelT##C3>::WarpPerspectiveBack(                                                          \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3, const PerspectiveTransformation<double> &aPerspective, \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C3 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//                                                                                                                       \
//    template void ImageView<pixelT##C2>::WarpPerspective(ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,       \
//                                                         ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2,       \
//                                                         const PerspectiveTransformation<double> &aPerspective,        \
//                                                         InterpolationMode aInterpolation, BorderType aBorder,         \
//                                                         Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx); \
//    template void ImageView<pixelT##C2>::WarpPerspective(                                                              \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2, const PerspectiveTransformation<double> &aPerspective, \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C2 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//    template void ImageView<pixelT##C2>::WarpPerspectiveBack(                                                          \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2, const PerspectiveTransformation<double> &aPerspective, \
//        InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi,                                     \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//    template void ImageView<pixelT##C2>::WarpPerspectiveBack(                                                          \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2, const PerspectiveTransformation<double> &aPerspective, \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C2 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateGeomTransform_For(pixelT) InstantiatePerspective_For(pixelT);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define ForAllChannelsNoAlpha(type)                                                                                    \
//    InstantiateGeomTransform_For(Pixel##type##C1);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C2);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C3);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4);                                                                     \
//    InstantiatePerspectivePlanar_For(Pixel##type);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define ForAllChannelsWithAlpha(type)                                                                                  \
//    InstantiateGeomTransform_For(Pixel##type##C1);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C2);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C3);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4A);                                                                    \
//    InstantiatePerspectivePlanar_For(Pixel##type);
//
//ForAllChannelsWithAlpha(8u);
//// ForAllChannelsWithAlpha(8s);
////
//// ForAllChannelsWithAlpha(16u);
//// ForAllChannelsWithAlpha(16s);
////
//// ForAllChannelsWithAlpha(32u);
//// ForAllChannelsWithAlpha(32s);
////
//// ForAllChannelsWithAlpha(16f);
//// ForAllChannelsWithAlpha(16bf);
//// ForAllChannelsWithAlpha(32f);
//// ForAllChannelsWithAlpha(64f);
////
//// ForAllChannelsNoAlpha(16sc);
//// ForAllChannelsNoAlpha(32sc);
//// ForAllChannelsNoAlpha(32fc);
//
//#undef InstantiatePerspectivePlanar_For
//#undef InstantiatePerspective_For
//#undef InstantiateGeomTransform_For
//#undef ForAllChannelsWithAlpha
//#undef ForAllChannelsNoAlpha
//#pragma endregion
//
//#pragma region Instantiate Rotate
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateRotate_For(pixelT)                                                                                  \
//    template ImageView<pixelT> &ImageView<pixelT>::Rotate(                                                             \
//        ImageView<pixelT> &aDst, double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,  \
//        BorderType aBorder, Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const;                        \
//    template ImageView<pixelT> &ImageView<pixelT>::Rotate(                                                             \
//        ImageView<pixelT> &aDst, double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,  \
//        BorderType aBorder, pixelT aConstant, Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const;
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateRotatePlanar_For(pixelT)                                                                            \
//    template void ImageView<pixelT##C4>::Rotate(ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4,                \
//                                                double aAngleInDeg, const Vector2<double> &aShift,                     \
//                                                InterpolationMode aInterpolation, BorderType aBorder,                  \
//                                                Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);          \
//    template void ImageView<pixelT##C4>::Rotate(                                                                       \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4, double aAngleInDeg, const Vector2<double> &aShift,     \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C4 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//                                                                                                                       \
//    template void ImageView<pixelT##C3>::Rotate(ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3,                \
//                                                double aAngleInDeg, const Vector2<double> &aShift,                     \
//                                                InterpolationMode aInterpolation, BorderType aBorder,                  \
//                                                Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);          \
//    template void ImageView<pixelT##C3>::Rotate(                                                                       \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3, double aAngleInDeg, const Vector2<double> &aShift,     \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C3 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
//                                                                                                                       \
//    template void ImageView<pixelT##C2>::Rotate(ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2,                \
//                                                double aAngleInDeg, const Vector2<double> &aShift,                     \
//                                                InterpolationMode aInterpolation, BorderType aBorder,                  \
//                                                Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);          \
//    template void ImageView<pixelT##C2>::Rotate(                                                                       \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2, double aAngleInDeg, const Vector2<double> &aShift,     \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT##C2 aConstant, Roi aAllowedReadRoi,               \
//        const mpp::cuda::StreamCtx &aStreamCtx);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateGeomTransform_For(pixelT) InstantiateRotate_For(pixelT);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define ForAllChannelsNoAlpha(type)                                                                                    \
//    InstantiateGeomTransform_For(Pixel##type##C1);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C2);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C3);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4);                                                                     \
//    InstantiateRotatePlanar_For(Pixel##type);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define ForAllChannelsWithAlpha(type)                                                                                  \
//    InstantiateGeomTransform_For(Pixel##type##C1);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C2);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C3);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4A);                                                                    \
//    InstantiateRotatePlanar_For(Pixel##type);
//
//ForAllChannelsWithAlpha(8u);
//// ForAllChannelsWithAlpha(8s);
////
//// ForAllChannelsWithAlpha(16u);
//// ForAllChannelsWithAlpha(16s);
////
//// ForAllChannelsWithAlpha(32u);
//// ForAllChannelsWithAlpha(32s);
////
//// ForAllChannelsWithAlpha(16f);
//// ForAllChannelsWithAlpha(16bf);
//// ForAllChannelsWithAlpha(32f);
//// ForAllChannelsWithAlpha(64f);
////
//// ForAllChannelsNoAlpha(16sc);
//// ForAllChannelsNoAlpha(32sc);
//// ForAllChannelsNoAlpha(32fc);
//
//#undef InstantiateRotatePlanar_For
//#undef InstantiateRotate_For
//#undef InstantiateGeomTransform_For
//#undef ForAllChannelsWithAlpha
//#undef ForAllChannelsNoAlpha
//#pragma endregion
//
//#pragma region Instantiate Resize
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateResize_For(pixelT)                                                                                  \
//    template ImageView<pixelT> &ImageView<pixelT>::Resize(                                                             \
//        ImageView<pixelT> &aDst, const Vector2<double> &aScale, const Vector2<double> &aShift,                         \
//        InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi,                                     \
//        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
//    template ImageView<pixelT> &ImageView<pixelT>::Resize(                                                             \
//        ImageView<pixelT> &aDst, const Vector2<double> &aScale, const Vector2<double> &aShift,                         \
//        InterpolationMode aInterpolation, BorderType aBorder, pixelT aConstant, Roi aAllowedReadRoi,                   \
//        const mpp::cuda::StreamCtx &aStreamCtx) const;
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateResizePlanar_For(pixelT)                                                                            \
//    template void ImageView<pixelT##C2>::Resize(ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2,                \
//                                                const Vector2<double> &aScale, const Vector2<double> &aShift,          \
//                                                InterpolationMode aInterpolation, BorderType aBorder,                  \
//                                                Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);          \
//    template void ImageView<pixelT##C2>::Resize(                                                                       \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C2>>> &aDst2, const Vector2<double> &aScale,                         \
//        const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder, pixelT##C2 aConstant,     \
//        Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);                                                  \
//                                                                                                                       \
//    template void ImageView<pixelT##C3>::Resize(ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3,                \
//                                                const Vector2<double> &aScale, const Vector2<double> &aShift,          \
//                                                InterpolationMode aInterpolation, BorderType aBorder,                  \
//                                                Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);          \
//    template void ImageView<pixelT##C3>::Resize(                                                                       \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C3>>> &aDst3, const Vector2<double> &aScale,                         \
//        const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder, pixelT##C3 aConstant,     \
//        Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);                                                  \
//                                                                                                                       \
//    template void ImageView<pixelT##C4>::Resize(ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,                \
//                                                ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4,                \
//                                                const Vector2<double> &aScale, const Vector2<double> &aShift,          \
//                                                InterpolationMode aInterpolation, BorderType aBorder,                  \
//                                                Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);          \
//    template void ImageView<pixelT##C4>::Resize(                                                                       \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aSrc4,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst1,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst2,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst3,                                                        \
//        ImageView<Vector1<remove_vector_t<pixelT##C4>>> &aDst4, const Vector2<double> &aScale,                         \
//        const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder, pixelT##C4 aConstant,     \
//        Roi aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define InstantiateGeomTransform_For(pixelT) InstantiateResize_For(pixelT);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define ForAllChannelsNoAlpha(type)                                                                                    \
//    InstantiateGeomTransform_For(Pixel##type##C1);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C2);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C3);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4);                                                                     \
//    InstantiateResizePlanar_For(Pixel##type);
//
//// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
//#define ForAllChannelsWithAlpha(type)                                                                                  \
//    InstantiateGeomTransform_For(Pixel##type##C1);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C2);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C3);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4);                                                                     \
//    InstantiateGeomTransform_For(Pixel##type##C4A);                                                                    \
//    InstantiateResizePlanar_For(Pixel##type);
//
//ForAllChannelsWithAlpha(8u);
//// ForAllChannelsWithAlpha(8s);
////
//// ForAllChannelsWithAlpha(16u);
//// ForAllChannelsWithAlpha(16s);
////
//// ForAllChannelsWithAlpha(32u);
//// ForAllChannelsWithAlpha(32s);
////
//// ForAllChannelsWithAlpha(16f);
//// ForAllChannelsWithAlpha(16bf);
//// ForAllChannelsWithAlpha(32f);
//// ForAllChannelsWithAlpha(64f);
////
//// ForAllChannelsNoAlpha(16sc);
//// ForAllChannelsNoAlpha(32sc);
//// ForAllChannelsNoAlpha(32fc);
//
//#undef InstantiateResizePlanar_For
//#undef InstantiateResize_For
//#undef InstantiateGeomTransform_For
//#undef ForAllChannelsWithAlpha
//#undef ForAllChannelsNoAlpha
//#pragma endregion
} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND