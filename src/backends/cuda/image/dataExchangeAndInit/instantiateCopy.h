#pragma once

namespace mpp::image::cuda
{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateCopy_For(type)                                                                                      \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C2> &ImageView<Pixel##type##C2>::Copy<Pixel##type##C2>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C2> & aDst, Channel aDstChannel,                                   \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C3> &ImageView<Pixel##type##C2>::Copy<Pixel##type##C3>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C3> & aDst, Channel aDstChannel,                                   \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C4> &ImageView<Pixel##type##C2>::Copy<Pixel##type##C4>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C4> & aDst, Channel aDstChannel,                                   \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
                                                                                                                       \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C2> &ImageView<Pixel##type##C3>::Copy<Pixel##type##C2>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C2> & aDst, Channel aDstChannel,                                   \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C3> &ImageView<Pixel##type##C3>::Copy<Pixel##type##C3>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C3> & aDst, Channel aDstChannel,                                   \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C4> &ImageView<Pixel##type##C3>::Copy<Pixel##type##C4>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C4> & aDst, Channel aDstChannel,                                   \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
                                                                                                                       \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C2> &ImageView<Pixel##type##C4>::Copy<Pixel##type##C2>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C2> & aDst, Channel aDstChannel,                                   \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C3> &ImageView<Pixel##type##C4>::Copy<Pixel##type##C3>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C3> & aDst, Channel aDstChannel,                                   \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C4> &ImageView<Pixel##type##C4>::Copy<Pixel##type##C4>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C4> & aDst, Channel aDstChannel,                                   \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
                                                                                                                       \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C2> &ImageView<Pixel##type##C1>::Copy<Pixel##type##C2>(            \
        ImageView<Pixel##type##C2> & aDst, Channel aDstChannel, const mpp::cuda::StreamCtx &aStreamCtx) const;         \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C3> &ImageView<Pixel##type##C1>::Copy<Pixel##type##C3>(            \
        ImageView<Pixel##type##C3> & aDst, Channel aDstChannel, const mpp::cuda::StreamCtx &aStreamCtx) const;         \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C4> &ImageView<Pixel##type##C1>::Copy<Pixel##type##C4>(            \
        ImageView<Pixel##type##C4> & aDst, Channel aDstChannel, const mpp::cuda::StreamCtx &aStreamCtx) const;         \
                                                                                                                       \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C1> &ImageView<Pixel##type##C2>::Copy<Pixel##type##C1>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C1> & aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;         \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C1> &ImageView<Pixel##type##C3>::Copy<Pixel##type##C1>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C1> & aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;         \
    template MPPEXPORT_CUDAI ImageView<Pixel##type##C1> &ImageView<Pixel##type##C4>::Copy<Pixel##type##C1>(            \
        Channel aSrcChannel, ImageView<Pixel##type##C1> & aDst, const mpp::cuda::StreamCtx &aStreamCtx) const;

} // namespace mpp::image::cuda