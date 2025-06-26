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

using Image8uC1View  = ImageView<Pixel8uC1>;
using Image8uC2View  = ImageView<Pixel8uC2>;
using Image8uC3View  = ImageView<Pixel8uC3>;
using Image8uC4View  = ImageView<Pixel8uC4>;
using Image8uC4AView = ImageView<Pixel8uC4A>;

template ImageView<Pixel32fC1> &ImageView<Pixel8uC1>::Convert<Pixel32fC1>(ImageView<Pixel32fC1> &aDst,
                                                                          const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel32fC3> &ImageView<Pixel8uC3>::Convert<Pixel32fC3>(ImageView<Pixel32fC3> &aDst,
                                                                          const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                         RoundingMode aRoundingMode,
                                                                         const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                         RoundingMode aRoundingMode, int aScaleFactor,
                                                                         const mpp::cuda::StreamCtx &aStreamCtx) const;

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

template class ImageView<Pixel16sC1>;
template <> ImageView<Pixel16sC1> ImageView<Pixel16sC1>::Null = ImageView<Pixel16sC1>(nullptr, Size2D(0, 0), 0);

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

using Image16uC1View  = ImageView<Pixel16uC1>;
using Image16uC2View  = ImageView<Pixel16uC2>;
using Image16uC3View  = ImageView<Pixel16uC3>;
using Image16uC4View  = ImageView<Pixel16uC4>;
using Image16uC4AView = ImageView<Pixel16uC4A>;

template ImageView<Pixel32fC3> &ImageView<Pixel16uC3>::Convert<Pixel32fC3>(
    ImageView<Pixel32fC3> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel32fC3>::Convert<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel32fC3>::Convert<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, RoundingMode aRoundingMode, int aScaleFactor,
    const mpp::cuda::StreamCtx &aStreamCtx) const;

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

template class ImageView<Pixel32fC1>;
template class ImageView<Pixel32fC2>;
template class ImageView<Pixel32fC3>;
template class ImageView<Pixel32fC4>;
template class ImageView<Pixel32fC4A>;

template <> ImageView<Pixel32fC1> ImageView<Pixel32fC1>::Null = ImageView<Pixel32fC1>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC2> ImageView<Pixel32fC2>::Null = ImageView<Pixel32fC2>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC3> ImageView<Pixel32fC3>::Null = ImageView<Pixel32fC3>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC4> ImageView<Pixel32fC4>::Null = ImageView<Pixel32fC4>(nullptr, Size2D(0, 0), 0);
template <> ImageView<Pixel32fC4A> ImageView<Pixel32fC4A>::Null = ImageView<Pixel32fC4A>(nullptr, Size2D(0, 0), 0);

using Image32fC1View  = ImageView<Pixel32fC1>;
using Image32fC2View  = ImageView<Pixel32fC2>;
using Image32fC3View  = ImageView<Pixel32fC3>;
using Image32fC4View  = ImageView<Pixel32fC4>;
using Image32fC4AView = ImageView<Pixel32fC4A>;

template class ImageView<Pixel32fcC1>;
using Image32fcC1View                             = ImageView<Pixel32fcC1>;
template <> ImageView<Pixel32fcC1> ImageView<Pixel32fcC1>::Null = ImageView<Pixel32fcC1>(nullptr, Size2D(0, 0), 0);


template class ImageView<Pixel32sC1>;
using Image32sC1View                                           = ImageView<Pixel32sC1>;
template <> ImageView<Pixel32sC1> ImageView<Pixel32sC1>::Null = ImageView<Pixel32sC1>(nullptr, Size2D(0, 0), 0);

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