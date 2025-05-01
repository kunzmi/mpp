#include "imageView.h"
#include "imageView_arithmetic_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_dataExchangeAndInit_impl.h" //NOLINT(misc-include-cleaner)
#include "imageView_geometryTransforms_impl.h"  //NOLINT(misc-include-cleaner)
#include "imageView_statistics_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_thresholdAndCompare_impl.h" //NOLINT(misc-include-cleaner)
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
#include <common/opp_defs.h> //NOLINT(misc-include-cleaner)

namespace opp::image::cpuSimple
{

using Image32fC1View  = ImageView<Pixel32fC1>;
using Image32fC2View  = ImageView<Pixel32fC2>;
using Image32fC3View  = ImageView<Pixel32fC3>;
using Image32fC4View  = ImageView<Pixel32fC4>;
using Image32fC4AView = ImageView<Pixel32fC4A>;

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

template ImageView<Pixel32fC3> &ImageView<Pixel8uC3>::Convert<Pixel32fC3>(ImageView<Pixel32fC3> &aDst) const;
template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                         RoundingMode aRoundingMode) const;
template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                         RoundingMode aRoundingMode,
                                                                         int aScaleFactor) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC2>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC2>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC2>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC3>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC3>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC3>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC4>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC4>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC4>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC1>::Copy<Pixel8uC2>(ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC1>::Copy<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC1>::Copy<Pixel8uC4>(ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel) const;

template ImageView<Pixel8uC1> &ImageView<Pixel8uC2>::Copy<Pixel8uC1>(Channel aSrcChannel,
                                                                     ImageView<Pixel8uC1> &aDst) const;
template ImageView<Pixel8uC1> &ImageView<Pixel8uC3>::Copy<Pixel8uC1>(Channel aSrcChannel,
                                                                     ImageView<Pixel8uC1> &aDst) const;
template ImageView<Pixel8uC1> &ImageView<Pixel8uC4>::Copy<Pixel8uC1>(Channel aSrcChannel,
                                                                     ImageView<Pixel8uC1> &aDst) const;

template ImageView<Pixel8uC2> &ImageView<Pixel8uC1>::Dup<Pixel8uC2>(ImageView<Pixel8uC2> &aDst) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC1>::Dup<Pixel8uC3>(ImageView<Pixel8uC3> &aDst) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC1>::Dup<Pixel8uC4>(ImageView<Pixel8uC4> &aDst) const;
template ImageView<Pixel8uC4A> &ImageView<Pixel8uC1>::Dup<Pixel8uC4A>(ImageView<Pixel8uC4A> &aDst) const;

template ImageView<Pixel8sC2> &ImageView<Pixel8sC1>::Dup<Pixel8sC2>(ImageView<Pixel8sC2> &aDst) const;
template ImageView<Pixel8sC3> &ImageView<Pixel8sC1>::Dup<Pixel8sC3>(ImageView<Pixel8sC3> &aDst) const;
template ImageView<Pixel8sC4> &ImageView<Pixel8sC1>::Dup<Pixel8sC4>(ImageView<Pixel8sC4> &aDst) const;
template ImageView<Pixel8sC4A> &ImageView<Pixel8sC1>::Dup<Pixel8sC4A>(ImageView<Pixel8sC4A> &aDst) const;

template ImageView<Pixel16uC2> &ImageView<Pixel16uC1>::Dup<Pixel16uC2>(ImageView<Pixel16uC2> &aDst) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC1>::Dup<Pixel16uC3>(ImageView<Pixel16uC3> &aDst) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC1>::Dup<Pixel16uC4>(ImageView<Pixel16uC4> &aDst) const;
template ImageView<Pixel16uC4A> &ImageView<Pixel16uC1>::Dup<Pixel16uC4A>(ImageView<Pixel16uC4A> &aDst) const;

template ImageView<Pixel16sC2> &ImageView<Pixel16sC1>::Dup<Pixel16sC2>(ImageView<Pixel16sC2> &aDst) const;
template ImageView<Pixel16sC3> &ImageView<Pixel16sC1>::Dup<Pixel16sC3>(ImageView<Pixel16sC3> &aDst) const;
template ImageView<Pixel16sC4> &ImageView<Pixel16sC1>::Dup<Pixel16sC4>(ImageView<Pixel16sC4> &aDst) const;
template ImageView<Pixel16sC4A> &ImageView<Pixel16sC1>::Dup<Pixel16sC4A>(ImageView<Pixel16sC4A> &aDst) const;

template ImageView<Pixel32uC2> &ImageView<Pixel32uC1>::Dup<Pixel32uC2>(ImageView<Pixel32uC2> &aDst) const;
template ImageView<Pixel32uC3> &ImageView<Pixel32uC1>::Dup<Pixel32uC3>(ImageView<Pixel32uC3> &aDst) const;
template ImageView<Pixel32uC4> &ImageView<Pixel32uC1>::Dup<Pixel32uC4>(ImageView<Pixel32uC4> &aDst) const;
template ImageView<Pixel32uC4A> &ImageView<Pixel32uC1>::Dup<Pixel32uC4A>(ImageView<Pixel32uC4A> &aDst) const;

template ImageView<Pixel32sC2> &ImageView<Pixel32sC1>::Dup<Pixel32sC2>(ImageView<Pixel32sC2> &aDst) const;
template ImageView<Pixel32sC3> &ImageView<Pixel32sC1>::Dup<Pixel32sC3>(ImageView<Pixel32sC3> &aDst) const;
template ImageView<Pixel32sC4> &ImageView<Pixel32sC1>::Dup<Pixel32sC4>(ImageView<Pixel32sC4> &aDst) const;
template ImageView<Pixel32sC4A> &ImageView<Pixel32sC1>::Dup<Pixel32sC4A>(ImageView<Pixel32sC4A> &aDst) const;

template ImageView<Pixel16fC2> &ImageView<Pixel16fC1>::Dup<Pixel16fC2>(ImageView<Pixel16fC2> &aDst) const;
template ImageView<Pixel16fC3> &ImageView<Pixel16fC1>::Dup<Pixel16fC3>(ImageView<Pixel16fC3> &aDst) const;
template ImageView<Pixel16fC4> &ImageView<Pixel16fC1>::Dup<Pixel16fC4>(ImageView<Pixel16fC4> &aDst) const;
template ImageView<Pixel16fC4A> &ImageView<Pixel16fC1>::Dup<Pixel16fC4A>(ImageView<Pixel16fC4A> &aDst) const;

template ImageView<Pixel16bfC2> &ImageView<Pixel16bfC1>::Dup<Pixel16bfC2>(ImageView<Pixel16bfC2> &aDst) const;
template ImageView<Pixel16bfC3> &ImageView<Pixel16bfC1>::Dup<Pixel16bfC3>(ImageView<Pixel16bfC3> &aDst) const;
template ImageView<Pixel16bfC4> &ImageView<Pixel16bfC1>::Dup<Pixel16bfC4>(ImageView<Pixel16bfC4> &aDst) const;
template ImageView<Pixel16bfC4A> &ImageView<Pixel16bfC1>::Dup<Pixel16bfC4A>(ImageView<Pixel16bfC4A> &aDst) const;

template ImageView<Pixel32fC2> &ImageView<Pixel32fC1>::Dup<Pixel32fC2>(ImageView<Pixel32fC2> &aDst) const;
template ImageView<Pixel32fC3> &ImageView<Pixel32fC1>::Dup<Pixel32fC3>(ImageView<Pixel32fC3> &aDst) const;
template ImageView<Pixel32fC4> &ImageView<Pixel32fC1>::Dup<Pixel32fC4>(ImageView<Pixel32fC4> &aDst) const;
template ImageView<Pixel32fC4A> &ImageView<Pixel32fC1>::Dup<Pixel32fC4A>(ImageView<Pixel32fC4A> &aDst) const;

template ImageView<Pixel64fC2> &ImageView<Pixel64fC1>::Dup<Pixel64fC2>(ImageView<Pixel64fC2> &aDst) const;
template ImageView<Pixel64fC3> &ImageView<Pixel64fC1>::Dup<Pixel64fC3>(ImageView<Pixel64fC3> &aDst) const;
template ImageView<Pixel64fC4> &ImageView<Pixel64fC1>::Dup<Pixel64fC4>(ImageView<Pixel64fC4> &aDst) const;
template ImageView<Pixel64fC4A> &ImageView<Pixel64fC1>::Dup<Pixel64fC4A>(ImageView<Pixel64fC4A> &aDst) const;

template ImageView<Pixel16scC2> &ImageView<Pixel16scC1>::Dup<Pixel16scC2>(ImageView<Pixel16scC2> &aDst) const;
template ImageView<Pixel16scC3> &ImageView<Pixel16scC1>::Dup<Pixel16scC3>(ImageView<Pixel16scC3> &aDst) const;
template ImageView<Pixel16scC4> &ImageView<Pixel16scC1>::Dup<Pixel16scC4>(ImageView<Pixel16scC4> &aDst) const;

template ImageView<Pixel32scC2> &ImageView<Pixel32scC1>::Dup<Pixel32scC2>(ImageView<Pixel32scC2> &aDst) const;
template ImageView<Pixel32scC3> &ImageView<Pixel32scC1>::Dup<Pixel32scC3>(ImageView<Pixel32scC3> &aDst) const;
template ImageView<Pixel32scC4> &ImageView<Pixel32scC1>::Dup<Pixel32scC4>(ImageView<Pixel32scC4> &aDst) const;

template ImageView<Pixel32fcC2> &ImageView<Pixel32fcC1>::Dup<Pixel32fcC2>(ImageView<Pixel32fcC2> &aDst) const;
template ImageView<Pixel32fcC3> &ImageView<Pixel32fcC1>::Dup<Pixel32fcC3>(ImageView<Pixel32fcC3> &aDst) const;
template ImageView<Pixel32fcC4> &ImageView<Pixel32fcC1>::Dup<Pixel32fcC4>(ImageView<Pixel32fcC4> &aDst) const;

template ImageView<Pixel8uC4> &ImageView<Pixel8uC3>::SwapChannel<Pixel8uC4>(
    ImageView<Pixel8uC4> &aDst, const ChannelList<vector_active_size_v<Pixel8uC4>> &aDstChannels,
    remove_vector_t<Pixel8uC3> aValue) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC4>::SwapChannel<Pixel8uC3>(
    ImageView<Pixel8uC3> &aDst, const ChannelList<vector_active_size_v<Pixel8uC3>> &aDstChannels) const;
template ImageView<Pixel8uC3> &ImageView<Pixel8uC3>::SwapChannel<Pixel8uC3>(
    ImageView<Pixel8uC3> &aDst, const ChannelList<vector_active_size_v<Pixel8uC3>> &aDstChannels) const;
template ImageView<Pixel8uC4> &ImageView<Pixel8uC4>::SwapChannel<Pixel8uC4>(
    ImageView<Pixel8uC4> &aDst, const ChannelList<vector_active_size_v<Pixel8uC4>> &aDstChannels) const;

template ImageView<Pixel8sC4> &ImageView<Pixel8sC3>::SwapChannel<Pixel8sC4>(
    ImageView<Pixel8sC4> &aDst, const ChannelList<vector_active_size_v<Pixel8sC4>> &aDstChannels,
    remove_vector_t<Pixel8sC3> aValue) const;
template ImageView<Pixel8sC3> &ImageView<Pixel8sC4>::SwapChannel<Pixel8sC3>(
    ImageView<Pixel8sC3> &aDst, const ChannelList<vector_active_size_v<Pixel8sC3>> &aDstChannels) const;
template ImageView<Pixel8sC3> &ImageView<Pixel8sC3>::SwapChannel<Pixel8sC3>(
    ImageView<Pixel8sC3> &aDst, const ChannelList<vector_active_size_v<Pixel8sC3>> &aDstChannels) const;
template ImageView<Pixel8sC4> &ImageView<Pixel8sC4>::SwapChannel<Pixel8sC4>(
    ImageView<Pixel8sC4> &aDst, const ChannelList<vector_active_size_v<Pixel8sC4>> &aDstChannels) const;

template ImageView<Pixel16uC4> &ImageView<Pixel16uC3>::SwapChannel<Pixel16uC4>(
    ImageView<Pixel16uC4> &aDst, const ChannelList<vector_active_size_v<Pixel16uC4>> &aDstChannels,
    remove_vector_t<Pixel16uC3> aValue) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC4>::SwapChannel<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, const ChannelList<vector_active_size_v<Pixel16uC3>> &aDstChannels) const;
template ImageView<Pixel16uC3> &ImageView<Pixel16uC3>::SwapChannel<Pixel16uC3>(
    ImageView<Pixel16uC3> &aDst, const ChannelList<vector_active_size_v<Pixel16uC3>> &aDstChannels) const;
template ImageView<Pixel16uC4> &ImageView<Pixel16uC4>::SwapChannel<Pixel16uC4>(
    ImageView<Pixel16uC4> &aDst, const ChannelList<vector_active_size_v<Pixel16uC4>> &aDstChannels) const;

template ImageView<Pixel16sC4> &ImageView<Pixel16sC3>::SwapChannel<Pixel16sC4>(
    ImageView<Pixel16sC4> &aDst, const ChannelList<vector_active_size_v<Pixel16sC4>> &aDstChannels,
    remove_vector_t<Pixel16sC3> aValue) const;
template ImageView<Pixel16sC3> &ImageView<Pixel16sC4>::SwapChannel<Pixel16sC3>(
    ImageView<Pixel16sC3> &aDst, const ChannelList<vector_active_size_v<Pixel16sC3>> &aDstChannels) const;
template ImageView<Pixel16sC3> &ImageView<Pixel16sC3>::SwapChannel<Pixel16sC3>(
    ImageView<Pixel16sC3> &aDst, const ChannelList<vector_active_size_v<Pixel16sC3>> &aDstChannels) const;
template ImageView<Pixel16sC4> &ImageView<Pixel16sC4>::SwapChannel<Pixel16sC4>(
    ImageView<Pixel16sC4> &aDst, const ChannelList<vector_active_size_v<Pixel16sC4>> &aDstChannels) const;

template ImageView<Pixel32uC4> &ImageView<Pixel32uC3>::SwapChannel<Pixel32uC4>(
    ImageView<Pixel32uC4> &aDst, const ChannelList<vector_active_size_v<Pixel32uC4>> &aDstChannels,
    remove_vector_t<Pixel32uC3> aValue) const;
template ImageView<Pixel32uC3> &ImageView<Pixel32uC4>::SwapChannel<Pixel32uC3>(
    ImageView<Pixel32uC3> &aDst, const ChannelList<vector_active_size_v<Pixel32uC3>> &aDstChannels) const;
template ImageView<Pixel32uC3> &ImageView<Pixel32uC3>::SwapChannel<Pixel32uC3>(
    ImageView<Pixel32uC3> &aDst, const ChannelList<vector_active_size_v<Pixel32uC3>> &aDstChannels) const;
template ImageView<Pixel32uC4> &ImageView<Pixel32uC4>::SwapChannel<Pixel32uC4>(
    ImageView<Pixel32uC4> &aDst, const ChannelList<vector_active_size_v<Pixel32uC4>> &aDstChannels) const;

template ImageView<Pixel32sC4> &ImageView<Pixel32sC3>::SwapChannel<Pixel32sC4>(
    ImageView<Pixel32sC4> &aDst, const ChannelList<vector_active_size_v<Pixel32sC4>> &aDstChannels,
    remove_vector_t<Pixel32sC3> aValue) const;
template ImageView<Pixel32sC3> &ImageView<Pixel32sC4>::SwapChannel<Pixel32sC3>(
    ImageView<Pixel32sC3> &aDst, const ChannelList<vector_active_size_v<Pixel32sC3>> &aDstChannels) const;
template ImageView<Pixel32sC3> &ImageView<Pixel32sC3>::SwapChannel<Pixel32sC3>(
    ImageView<Pixel32sC3> &aDst, const ChannelList<vector_active_size_v<Pixel32sC3>> &aDstChannels) const;
template ImageView<Pixel32sC4> &ImageView<Pixel32sC4>::SwapChannel<Pixel32sC4>(
    ImageView<Pixel32sC4> &aDst, const ChannelList<vector_active_size_v<Pixel32sC4>> &aDstChannels) const;

template ImageView<Pixel16fC4> &ImageView<Pixel16fC3>::SwapChannel<Pixel16fC4>(
    ImageView<Pixel16fC4> &aDst, const ChannelList<vector_active_size_v<Pixel16fC4>> &aDstChannels,
    remove_vector_t<Pixel16fC3> aValue) const;
template ImageView<Pixel16fC3> &ImageView<Pixel16fC4>::SwapChannel<Pixel16fC3>(
    ImageView<Pixel16fC3> &aDst, const ChannelList<vector_active_size_v<Pixel16fC3>> &aDstChannels) const;
template ImageView<Pixel16fC3> &ImageView<Pixel16fC3>::SwapChannel<Pixel16fC3>(
    ImageView<Pixel16fC3> &aDst, const ChannelList<vector_active_size_v<Pixel16fC3>> &aDstChannels) const;
template ImageView<Pixel16fC4> &ImageView<Pixel16fC4>::SwapChannel<Pixel16fC4>(
    ImageView<Pixel16fC4> &aDst, const ChannelList<vector_active_size_v<Pixel16fC4>> &aDstChannels) const;

template ImageView<Pixel16bfC4> &ImageView<Pixel16bfC3>::SwapChannel<Pixel16bfC4>(
    ImageView<Pixel16bfC4> &aDst, const ChannelList<vector_active_size_v<Pixel16bfC4>> &aDstChannels,
    remove_vector_t<Pixel16bfC3> aValue) const;
template ImageView<Pixel16bfC3> &ImageView<Pixel16bfC4>::SwapChannel<Pixel16bfC3>(
    ImageView<Pixel16bfC3> &aDst, const ChannelList<vector_active_size_v<Pixel16bfC3>> &aDstChannels) const;
template ImageView<Pixel16bfC3> &ImageView<Pixel16bfC3>::SwapChannel<Pixel16bfC3>(
    ImageView<Pixel16bfC3> &aDst, const ChannelList<vector_active_size_v<Pixel16bfC3>> &aDstChannels) const;
template ImageView<Pixel16bfC4> &ImageView<Pixel16bfC4>::SwapChannel<Pixel16bfC4>(
    ImageView<Pixel16bfC4> &aDst, const ChannelList<vector_active_size_v<Pixel16bfC4>> &aDstChannels) const;

template ImageView<Pixel32fC4> &ImageView<Pixel32fC3>::SwapChannel<Pixel32fC4>(
    ImageView<Pixel32fC4> &aDst, const ChannelList<vector_active_size_v<Pixel32fC4>> &aDstChannels,
    remove_vector_t<Pixel32fC3> aValue) const;
template ImageView<Pixel32fC3> &ImageView<Pixel32fC4>::SwapChannel<Pixel32fC3>(
    ImageView<Pixel32fC3> &aDst, const ChannelList<vector_active_size_v<Pixel32fC3>> &aDstChannels) const;
template ImageView<Pixel32fC3> &ImageView<Pixel32fC3>::SwapChannel<Pixel32fC3>(
    ImageView<Pixel32fC3> &aDst, const ChannelList<vector_active_size_v<Pixel32fC3>> &aDstChannels) const;
template ImageView<Pixel32fC4> &ImageView<Pixel32fC4>::SwapChannel<Pixel32fC4>(
    ImageView<Pixel32fC4> &aDst, const ChannelList<vector_active_size_v<Pixel32fC4>> &aDstChannels) const;

template ImageView<Pixel64fC4> &ImageView<Pixel64fC3>::SwapChannel<Pixel64fC4>(
    ImageView<Pixel64fC4> &aDst, const ChannelList<vector_active_size_v<Pixel64fC4>> &aDstChannels,
    remove_vector_t<Pixel64fC3> aValue) const;
template ImageView<Pixel64fC3> &ImageView<Pixel64fC4>::SwapChannel<Pixel64fC3>(
    ImageView<Pixel64fC3> &aDst, const ChannelList<vector_active_size_v<Pixel64fC3>> &aDstChannels) const;
template ImageView<Pixel64fC3> &ImageView<Pixel64fC3>::SwapChannel<Pixel64fC3>(
    ImageView<Pixel64fC3> &aDst, const ChannelList<vector_active_size_v<Pixel64fC3>> &aDstChannels) const;
template ImageView<Pixel64fC4> &ImageView<Pixel64fC4>::SwapChannel<Pixel64fC4>(
    ImageView<Pixel64fC4> &aDst, const ChannelList<vector_active_size_v<Pixel64fC4>> &aDstChannels) const;

template ImageView<Pixel16scC4> &ImageView<Pixel16scC3>::SwapChannel<Pixel16scC4>(
    ImageView<Pixel16scC4> &aDst, const ChannelList<vector_active_size_v<Pixel16scC4>> &aDstChannels,
    remove_vector_t<Pixel16scC3> aValue) const;
template ImageView<Pixel16scC3> &ImageView<Pixel16scC4>::SwapChannel<Pixel16scC3>(
    ImageView<Pixel16scC3> &aDst, const ChannelList<vector_active_size_v<Pixel16scC3>> &aDstChannels) const;
template ImageView<Pixel16scC3> &ImageView<Pixel16scC3>::SwapChannel<Pixel16scC3>(
    ImageView<Pixel16scC3> &aDst, const ChannelList<vector_active_size_v<Pixel16scC3>> &aDstChannels) const;
template ImageView<Pixel16scC4> &ImageView<Pixel16scC4>::SwapChannel<Pixel16scC4>(
    ImageView<Pixel16scC4> &aDst, const ChannelList<vector_active_size_v<Pixel16scC4>> &aDstChannels) const;

template ImageView<Pixel32scC4> &ImageView<Pixel32scC3>::SwapChannel<Pixel32scC4>(
    ImageView<Pixel32scC4> &aDst, const ChannelList<vector_active_size_v<Pixel32scC4>> &aDstChannels,
    remove_vector_t<Pixel32scC3> aValue) const;
template ImageView<Pixel32scC3> &ImageView<Pixel32scC4>::SwapChannel<Pixel32scC3>(
    ImageView<Pixel32scC3> &aDst, const ChannelList<vector_active_size_v<Pixel32scC3>> &aDstChannels) const;
template ImageView<Pixel32scC3> &ImageView<Pixel32scC3>::SwapChannel<Pixel32scC3>(
    ImageView<Pixel32scC3> &aDst, const ChannelList<vector_active_size_v<Pixel32scC3>> &aDstChannels) const;
template ImageView<Pixel32scC4> &ImageView<Pixel32scC4>::SwapChannel<Pixel32scC4>(
    ImageView<Pixel32scC4> &aDst, const ChannelList<vector_active_size_v<Pixel32scC4>> &aDstChannels) const;

template ImageView<Pixel32fcC4> &ImageView<Pixel32fcC3>::SwapChannel<Pixel32fcC4>(
    ImageView<Pixel32fcC4> &aDst, const ChannelList<vector_active_size_v<Pixel32fcC4>> &aDstChannels,
    remove_vector_t<Pixel32fcC3> aValue) const;
template ImageView<Pixel32fcC3> &ImageView<Pixel32fcC4>::SwapChannel<Pixel32fcC3>(
    ImageView<Pixel32fcC3> &aDst, const ChannelList<vector_active_size_v<Pixel32fcC3>> &aDstChannels) const;
template ImageView<Pixel32fcC3> &ImageView<Pixel32fcC3>::SwapChannel<Pixel32fcC3>(
    ImageView<Pixel32fcC3> &aDst, const ChannelList<vector_active_size_v<Pixel32fcC3>> &aDstChannels) const;
template ImageView<Pixel32fcC4> &ImageView<Pixel32fcC4>::SwapChannel<Pixel32fcC4>(
    ImageView<Pixel32fcC4> &aDst, const ChannelList<vector_active_size_v<Pixel32fcC4>> &aDstChannels) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateAffine_For(pixelT, coordT)                                                                          \
    template ImageView<pixelT> &ImageView<pixelT>::WarpAffine(                                                         \
        ImageView<pixelT> &aDst, const AffineTransformation<coordT> &aAffine, InterpolationMode aInterpolation,        \
        BorderType aBorder, Roi aAllowedReadRoi) const;                                                                \
    template ImageView<pixelT> &ImageView<pixelT>::WarpAffine(                                                         \
        ImageView<pixelT> &aDst, const AffineTransformation<coordT> &aAffine, InterpolationMode aInterpolation,        \
        BorderType aBorder, pixelT aConstant, Roi aAllowedReadRoi) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsNoAlpha(type)                                                                                    \
    InstantiateAffine_For(Pixel##type##C1, float);                                                                     \
    InstantiateAffine_For(Pixel##type##C2, float);                                                                     \
    InstantiateAffine_For(Pixel##type##C3, float);                                                                     \
    InstantiateAffine_For(Pixel##type##C4, float);                                                                     \
    InstantiateAffine_For(Pixel##type##C1, double);                                                                    \
    InstantiateAffine_For(Pixel##type##C2, double);                                                                    \
    InstantiateAffine_For(Pixel##type##C3, double);                                                                    \
    InstantiateAffine_For(Pixel##type##C4, double);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ForAllChannelsWithAlpha(type)                                                                                  \
    InstantiateAffine_For(Pixel##type##C1, float);                                                                     \
    InstantiateAffine_For(Pixel##type##C2, float);                                                                     \
    InstantiateAffine_For(Pixel##type##C3, float);                                                                     \
    InstantiateAffine_For(Pixel##type##C4, float);                                                                     \
    InstantiateAffine_For(Pixel##type##C4A, float);                                                                    \
    InstantiateAffine_For(Pixel##type##C1, double);                                                                    \
    InstantiateAffine_For(Pixel##type##C2, double);                                                                    \
    InstantiateAffine_For(Pixel##type##C3, double);                                                                    \
    InstantiateAffine_For(Pixel##type##C4, double);                                                                    \
    InstantiateAffine_For(Pixel##type##C4A, double);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);
#undef InstantiateAffine_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha

} // namespace opp::image::cpuSimple