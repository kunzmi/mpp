#pragma once

namespace mpp::image::cpuSimple
{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define InstantiateSwapChannel_For(type)                                                                               \
    template ImageView<Pixel##type##C4> &ImageView<Pixel##type##C3>::SwapChannel<Pixel##type##C4>(                     \
        ImageView<Pixel##type##C4> & aDst, const ChannelList<vector_active_size_v<Pixel##type##C4>> &aDstChannels,     \
        remove_vector_t<Pixel##type##C3> aValue) const;                                                                \
    template ImageView<Pixel##type##C3> &ImageView<Pixel##type##C4>::SwapChannel<Pixel##type##C3>(                     \
        ImageView<Pixel##type##C3> & aDst, const ChannelList<vector_active_size_v<Pixel##type##C3>> &aDstChannels)     \
        const;                                                                                                         \
    template ImageView<Pixel##type##C3> &ImageView<Pixel##type##C3>::SwapChannel<Pixel##type##C3>(                     \
        ImageView<Pixel##type##C3> & aDst, const ChannelList<vector_active_size_v<Pixel##type##C3>> &aDstChannels)     \
        const;                                                                                                         \
    template ImageView<Pixel##type##C4> &ImageView<Pixel##type##C4>::SwapChannel<Pixel##type##C4>(                     \
        ImageView<Pixel##type##C4> & aDst, const ChannelList<vector_active_size_v<Pixel##type##C4>> &aDstChannels)     \
        const;

} // namespace mpp::image::cpuSimple