#pragma once

namespace mpp::image::cuda
{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateSwapChannel_For(type)                                                                               \
    template ImageView<Pixel##type##C4> &ImageView<Pixel##type##C3>::SwapChannel<Pixel##type##C4>(                     \
        ImageView<Pixel##type##C4> & aDst, const ChannelList<vector_active_size_v<Pixel##type##C4>> &aDstChannels,     \
        remove_vector_t<Pixel##type##C3> aValue, const mpp::cuda::StreamCtx &aStreamCtx) const;                        \
    template ImageView<Pixel##type##C3> &ImageView<Pixel##type##C4>::SwapChannel<Pixel##type##C3>(                     \
        ImageView<Pixel##type##C3> & aDst, const ChannelList<vector_active_size_v<Pixel##type##C3>> &aDstChannels,     \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
    template ImageView<Pixel##type##C3> &ImageView<Pixel##type##C3>::SwapChannel<Pixel##type##C3>(                     \
        ImageView<Pixel##type##C3> & aDst, const ChannelList<vector_active_size_v<Pixel##type##C3>> &aDstChannels,     \
        const mpp::cuda::StreamCtx &aStreamCtx) const;                                                                 \
    template ImageView<Pixel##type##C4> &ImageView<Pixel##type##C4>::SwapChannel<Pixel##type##C4>(                     \
        ImageView<Pixel##type##C4> & aDst, const ChannelList<vector_active_size_v<Pixel##type##C4>> &aDstChannels,     \
        const mpp::cuda::StreamCtx &aStreamCtx) const;

} // namespace mpp::image::cuda