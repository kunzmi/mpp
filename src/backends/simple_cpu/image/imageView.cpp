#include "conversionRelations.h"
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

// template ImageView<Pixel16uC1> &ImageView<Pixel8uC1>::Convert<Pixel16uC1>(ImageView<Pixel16uC1> &aDst) const;
// template ImageView<Pixel32sC1> &ImageView<Pixel8uC1>::Convert<Pixel32sC1>(ImageView<Pixel32sC1> &aDst) const;
// template ImageView<Pixel32fC1> &ImageView<Pixel8uC1>::Convert<Pixel32fC1>(ImageView<Pixel32fC1> &aDst) const;
// template ImageView<Pixel64fC1> &ImageView<Pixel8uC1>::Convert<Pixel64fC1>(ImageView<Pixel64fC1> &aDst) const;
// template ImageView<Pixel32fC3> &ImageView<Pixel8uC3>::Convert<Pixel32fC3>(ImageView<Pixel32fC3> &aDst) const;
// template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
//                                                                          RoundingMode aRoundingMode) const;
// template ImageView<Pixel8uC3> &ImageView<Pixel32fC3>::Convert<Pixel8uC3>(ImageView<Pixel8uC3> &aDst,
//                                                                          RoundingMode aRoundingMode,
//                                                                          int aScaleFactor) const;

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

template ImageView<Pixel16uC1> &ImageView<Pixel8uC1>::Scale<Pixel16uC1>(ImageView<Pixel16uC1> &aDst) const;
template ImageView<Pixel16sC1> &ImageView<Pixel8uC1>::Scale<Pixel16sC1>(ImageView<Pixel16sC1> &aDst) const;
template ImageView<Pixel32uC1> &ImageView<Pixel8uC1>::Scale<Pixel32uC1>(ImageView<Pixel32uC1> &aDst) const;
template ImageView<Pixel32sC1> &ImageView<Pixel8uC1>::Scale<Pixel32sC1>(ImageView<Pixel32sC1> &aDst) const;
template ImageView<Pixel16uC1> &ImageView<Pixel8uC1>::Scale<Pixel16uC1>(ImageView<Pixel16uC1> &aDst, byte, byte) const;
template ImageView<Pixel16sC1> &ImageView<Pixel8uC1>::Scale<Pixel16sC1>(ImageView<Pixel16sC1> &aDst, byte, byte) const;
template ImageView<Pixel32uC1> &ImageView<Pixel8uC1>::Scale<Pixel32uC1>(ImageView<Pixel32uC1> &aDst, byte, byte) const;
template ImageView<Pixel32sC1> &ImageView<Pixel8uC1>::Scale<Pixel32sC1>(ImageView<Pixel32sC1> &aDst, byte, byte) const;
template ImageView<Pixel16uC1> &ImageView<Pixel8uC1>::Scale<Pixel16uC1>(ImageView<Pixel16uC1> &aDst, ushort,
                                                                        ushort) const;
template ImageView<Pixel16sC1> &ImageView<Pixel8uC1>::Scale<Pixel16sC1>(ImageView<Pixel16sC1> &aDst, short,
                                                                        short) const;
template ImageView<Pixel32uC1> &ImageView<Pixel8uC1>::Scale<Pixel32uC1>(ImageView<Pixel32uC1> &aDst, uint, uint) const;
template ImageView<Pixel32sC1> &ImageView<Pixel8uC1>::Scale<Pixel32sC1>(ImageView<Pixel32sC1> &aDst, int, int) const;
template ImageView<Pixel32fC1> &ImageView<Pixel8uC1>::Scale<Pixel32fC1>(ImageView<Pixel32fC1> &aDst, float,
                                                                        float) const;
template ImageView<Pixel16uC1> &ImageView<Pixel8uC1>::Scale<Pixel16uC1>(ImageView<Pixel16uC1> &aDst, byte, byte, ushort,
                                                                        ushort) const;
template ImageView<Pixel16sC1> &ImageView<Pixel8uC1>::Scale<Pixel16sC1>(ImageView<Pixel16sC1> &aDst, byte, byte, short,
                                                                        short) const;
template ImageView<Pixel32uC1> &ImageView<Pixel8uC1>::Scale<Pixel32uC1>(ImageView<Pixel32uC1> &aDst, byte, byte, uint,
                                                                        uint) const;
template ImageView<Pixel32sC1> &ImageView<Pixel8uC1>::Scale<Pixel32sC1>(ImageView<Pixel32sC1> &aDst, byte, byte, int,
                                                                        int) const;
template ImageView<Pixel32fC1> &ImageView<Pixel8uC1>::Scale<Pixel32fC1>(ImageView<Pixel32fC1> &aDst, byte, byte, float,
                                                                        float) const;

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

} // namespace mpp::image::cpuSimple