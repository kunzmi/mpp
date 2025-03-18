#include "imageView.h"
#include "imageView_arithmetic_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_dataExchangeAndInit_impl.h" //NOLINT(misc-include-cleaner)
#include "imageView_statistics_impl.h"          //NOLINT(misc-include-cleaner)
#include "imageView_thresholdAndCompare_impl.h" //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixel_impl.h>              //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/forEachPixelMasked_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixelPlanar.h>
#include <backends/simple_cpu/image/forEachPixelPlanar_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/simple_cpu/image/forEachPixelSingleChannel.h>
#include <backends/simple_cpu/image/forEachPixelSingleChannel_impl.h> //NOLINT(misc-include-cleaner)
#include <common/image/pixelTypes.h>
#include <common/opp_defs.h> //NOLINT(misc-include-cleaner)

namespace opp::image::cpuSimple
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

template class ImageView<Pixel32fC1>;
template class ImageView<Pixel32fC2>;
template class ImageView<Pixel32fC3>;
template class ImageView<Pixel32fC4>;
template class ImageView<Pixel32fC4A>;

template class ImageView<Pixel32fcC1>;
template class ImageView<Pixel32fcC2>;
template class ImageView<Pixel32fcC3>;
template class ImageView<Pixel32fcC4>;

template ImageView<Pixel32fC3> &ImageView<Pixel8uC3>::Convert<Pixel32fC3>(ImageView<Pixel32fC3> &aDst);
template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                         RoundingMode aRoundingMode);
template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
                                                                         RoundingMode aRoundingMode, int aScaleFactor);

template ImageView<Pixel8uC2> &ImageView<Pixel8uC2>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel);
template ImageView<Pixel8uC3> &ImageView<Pixel8uC2>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel);
template ImageView<Pixel8uC4> &ImageView<Pixel8uC2>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel);

template ImageView<Pixel8uC2> &ImageView<Pixel8uC3>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel);
template ImageView<Pixel8uC3> &ImageView<Pixel8uC3>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel);
template ImageView<Pixel8uC4> &ImageView<Pixel8uC3>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel);

template ImageView<Pixel8uC2> &ImageView<Pixel8uC4>::Copy<Pixel8uC2>(Channel aSrcChannel, ImageView<Pixel8uC2> &aDst,
                                                                     Channel aDstChannel);
template ImageView<Pixel8uC3> &ImageView<Pixel8uC4>::Copy<Pixel8uC3>(Channel aSrcChannel, ImageView<Pixel8uC3> &aDst,
                                                                     Channel aDstChannel);
template ImageView<Pixel8uC4> &ImageView<Pixel8uC4>::Copy<Pixel8uC4>(Channel aSrcChannel, ImageView<Pixel8uC4> &aDst,
                                                                     Channel aDstChannel);

template ImageView<Pixel8uC2> &ImageView<Pixel8uC1>::Copy<Pixel8uC2>(ImageView<Pixel8uC2> &aDst, Channel aDstChannel);
template ImageView<Pixel8uC3> &ImageView<Pixel8uC1>::Copy<Pixel8uC3>(ImageView<Pixel8uC3> &aDst, Channel aDstChannel);
template ImageView<Pixel8uC4> &ImageView<Pixel8uC1>::Copy<Pixel8uC4>(ImageView<Pixel8uC4> &aDst, Channel aDstChannel);

template ImageView<Pixel8uC1> &ImageView<Pixel8uC2>::Copy<Pixel8uC1>(Channel aSrcChannel, ImageView<Pixel8uC1> &aDst);
template ImageView<Pixel8uC1> &ImageView<Pixel8uC3>::Copy<Pixel8uC1>(Channel aSrcChannel, ImageView<Pixel8uC1> &aDst);
template ImageView<Pixel8uC1> &ImageView<Pixel8uC4>::Copy<Pixel8uC1>(Channel aSrcChannel, ImageView<Pixel8uC1> &aDst);

template ImageView<Pixel8uC3> &ImageView<Pixel8uC1>::Dup<Pixel8uC3>(ImageView<Pixel8uC3> &aDst);
template ImageView<Pixel8uC4> &ImageView<Pixel8uC1>::Dup<Pixel8uC4>(ImageView<Pixel8uC4> &aDst);
template ImageView<Pixel8uC4A> &ImageView<Pixel8uC1>::Dup<Pixel8uC4A>(ImageView<Pixel8uC4A> &aDst);

template ImageView<Pixel8uC4> &ImageView<Pixel8uC3>::SwapChannel<Pixel8uC4>(
    ImageView<Pixel8uC4> &aDst, const ChannelList<vector_active_size_v<Pixel8uC4>> &aDstChannels,
    remove_vector_t<Pixel8uC3> aValue);
template ImageView<Pixel8uC3> &ImageView<Pixel8uC4>::SwapChannel<Pixel8uC3>(
    ImageView<Pixel8uC3> &aDst, const ChannelList<vector_active_size_v<Pixel8uC3>> &aDstChannels);
template ImageView<Pixel8uC3> &ImageView<Pixel8uC3>::SwapChannel<Pixel8uC3>(
    ImageView<Pixel8uC3> &aDst, const ChannelList<vector_active_size_v<Pixel8uC3>> &aDstChannels);
template ImageView<Pixel8uC4> &ImageView<Pixel8uC4>::SwapChannel<Pixel8uC4>(
    ImageView<Pixel8uC4> &aDst, const ChannelList<vector_active_size_v<Pixel8uC4>> &aDstChannels);
template ImageView<Pixel8uC4A> &ImageView<Pixel8uC4A>::SwapChannel<Pixel8uC4A>(
    ImageView<Pixel8uC4A> &aDst, const ChannelList<vector_active_size_v<Pixel8uC4A>> &aDstChannels);

using Image32fC1View  = ImageView<Pixel32fC1>;
using Image32fC2View  = ImageView<Pixel32fC2>;
using Image32fC3View  = ImageView<Pixel32fC3>;
using Image32fC4View  = ImageView<Pixel32fC4>;
using Image32fC4AView = ImageView<Pixel32fC4A>;

} // namespace opp::image::cpuSimple