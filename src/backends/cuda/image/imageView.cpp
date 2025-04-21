#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "imageView.h"
#include "imageView_arithmetic_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_dataExchangeAndInit_impl.h" //NOLINT(misc-include-cleaner)
#include "imageView_statistics_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_thresholdAndCompare_impl.h" //NOLINT(misc-include-cleaner)
#include <backends/cuda/streamCtx.h>            //NOLINT(misc-include-cleaner)
#include <common/image/pixelTypes.h>
#include <common/opp_defs.h> //NOLINT(misc-include-cleaner)

namespace opp::image::cuda
{
template class ImageView<Pixel8uC1>;
template class ImageView<Pixel8uC2>;
template class ImageView<Pixel8uC3>;
template class ImageView<Pixel8uC4>;
template class ImageView<Pixel8uC4A>;

using Image8uC1View  = ImageView<Pixel8uC1>;
using Image8uC2View  = ImageView<Pixel8uC2>;
using Image8uC3View  = ImageView<Pixel8uC3>;
using Image8uC4View  = ImageView<Pixel8uC4>;
using Image8uC4AView = ImageView<Pixel8uC4A>;

template ImageView<Pixel32fC3> &ImageView<Pixel8uC3>::Convert<Pixel32fC3>(ImageView<Pixel32fC3> &aDst,
                                                                          const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                         RoundingMode aRoundingMode,
                                                                         const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                         RoundingMode aRoundingMode, int aScaleFactor,
                                                                         const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC2>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC2>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC2>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC3>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC3>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC3>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC4>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC4>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC4>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC1>::Copy<Pixel8uC2>(ImageView<Pixel8uC2> &aDst, Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC1>::Copy<Pixel8uC3>(ImageView<Pixel8uC3> &aDst, Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC1>::Copy<Pixel8uC4>(ImageView<Pixel8uC4> &aDst, Channel aDstChannel,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC1> &ImageView<Pixel8uC2>::Copy<Pixel8uC1>(Channel aSrcChannel, ImageView<Pixel8uC1> &aDst,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC1> &ImageView<Pixel8uC3>::Copy<Pixel8uC1>(Channel aSrcChannel, ImageView<Pixel8uC1> &aDst,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC1> &ImageView<Pixel8uC4>::Copy<Pixel8uC1>(Channel aSrcChannel, ImageView<Pixel8uC1> &aDst,
                                                                     const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC3> &ImageView<Pixel8uC1>::Dup<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                    const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC1>::Dup<Pixel8uC4>(ImageView<Pixel8uC4> &aDst,
                                                                    const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4A> &ImageView<Pixel8uC1>::Dup<Pixel8uC4A>(ImageView<Pixel8uC4A> &aDst,
                                                                      const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel8uC4> &ImageView<Pixel8uC3>::SwapChannel<Pixel8uC4>(
    ImageView<Pixel8uC4> &aDst, const ChannelList<vector_active_size_v<Pixel8uC4>> &aDstChannels,
    remove_vector_t<Pixel8uC3> aValue, const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC4>::SwapChannel<Pixel8uC3>(
    ImageView<Pixel8uC3> &aDst, const ChannelList<vector_active_size_v<Pixel8uC3>> &aDstChannels,
    const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC3>::SwapChannel<Pixel8uC3>(
    ImageView<Pixel8uC3> &aDst, const ChannelList<vector_active_size_v<Pixel8uC3>> &aDstChannels,
    const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC4>::SwapChannel<Pixel8uC4>(
    ImageView<Pixel8uC4> &aDst, const ChannelList<vector_active_size_v<Pixel8uC4>> &aDstChannels,
    const opp::cuda::StreamCtx &aStreamCtx) const;

template class ImageView<Pixel16uC1>;
template class ImageView<Pixel16uC2>;
template class ImageView<Pixel16uC3>;
template class ImageView<Pixel16uC4>;
template class ImageView<Pixel16uC4A>;

using Image16uC1View  = ImageView<Pixel16uC1>;
using Image16uC2View  = ImageView<Pixel16uC2>;
using Image16uC3View  = ImageView<Pixel16uC3>;
using Image16uC4View  = ImageView<Pixel16uC4>;
using Image16uC4AView = ImageView<Pixel16uC4A>;

template ImageView<Pixel32fC3> &ImageView<Pixel16uC3>::Convert<Pixel32fC3>(
    ImageView<Pixel32fC3> &aDst, const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel32fC3>::Convert<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel32fC3>::Convert<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, RoundingMode aRoundingMode, int aScaleFactor,
    const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC2> &ImageView<Pixel16uC2>::Copy<Pixel16uC2>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC2> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC2>::Copy<Pixel16uC3>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC3> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC2>::Copy<Pixel16uC4>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC4> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC2> &ImageView<Pixel16uC3>::Copy<Pixel16uC2>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC2> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC3>::Copy<Pixel16uC3>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC3> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC3>::Copy<Pixel16uC4>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC4> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC2> &ImageView<Pixel16uC4>::Copy<Pixel16uC2>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC2> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC4>::Copy<Pixel16uC3>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC3> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC4>::Copy<Pixel16uC4>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC4> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC2> &ImageView<Pixel16uC1>::Copy<Pixel16uC2>(ImageView<Pixel16uC2> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC1>::Copy<Pixel16uC3>(ImageView<Pixel16uC3> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC1>::Copy<Pixel16uC4>(ImageView<Pixel16uC4> &aDst,
                                                                        Channel aDstChannel,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC1> &ImageView<Pixel16uC2>::Copy<Pixel16uC1>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC1> &aDst,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC1> &ImageView<Pixel16uC3>::Copy<Pixel16uC1>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC1> &aDst,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC1> &ImageView<Pixel16uC4>::Copy<Pixel16uC1>(Channel aSrcChannel,
                                                                        ImageView<Pixel16uC1> &aDst,
                                                                        const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC3> &ImageView<Pixel16uC1>::Dup<Pixel16uC3>(ImageView<Pixel16uC3> &aDst,
                                                                       const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC1>::Dup<Pixel16uC4>(ImageView<Pixel16uC4> &aDst,
                                                                       const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4A> &ImageView<Pixel16uC1>::Dup<Pixel16uC4A>(ImageView<Pixel16uC4A> &aDst,
                                                                         const opp::cuda::StreamCtx &aStreamCtx) const;

template ImageView<Pixel16uC4> &ImageView<Pixel16uC3>::SwapChannel<Pixel16uC4>(
    ImageView<Pixel16uC4> &aDst, const ChannelList<vector_active_size_v<Pixel16uC4>> &aDstChannels,
    remove_vector_t<Pixel16uC3> aValue, const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC4>::SwapChannel<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, const ChannelList<vector_active_size_v<Pixel16uC3>> &aDstChannels,
    const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC3>::SwapChannel<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, const ChannelList<vector_active_size_v<Pixel16uC3>> &aDstChannels,
    const opp::cuda::StreamCtx &aStreamCtx) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC4>::SwapChannel<Pixel16uC4>(
    ImageView<Pixel16uC4> &aDst, const ChannelList<vector_active_size_v<Pixel16uC4>> &aDstChannels,
    const opp::cuda::StreamCtx &aStreamCtx) const;

template class ImageView<Pixel32fC1>;
template class ImageView<Pixel32fC2>;
template class ImageView<Pixel32fC3>;
template class ImageView<Pixel32fC4>;
template class ImageView<Pixel32fC4A>;

using Image32fC1View  = ImageView<Pixel32fC1>;
using Image32fC2View  = ImageView<Pixel32fC2>;
using Image32fC3View  = ImageView<Pixel32fC3>;
using Image32fC4View  = ImageView<Pixel32fC4>;
using Image32fC4AView = ImageView<Pixel32fC4A>;

template class ImageView<Pixel32fcC1>;
using Image32fcC1View = ImageView<Pixel32fcC1>;
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND