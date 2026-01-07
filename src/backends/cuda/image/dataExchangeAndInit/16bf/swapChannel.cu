#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16bf);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16bf);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16bf);
ForAllChannelsNoAlphaInvokeSwapChannelSrc2(16bf);
ForAllChannelsNoAlphaInvokeSwapChannelInplace2(16bf);

} // namespace mpp::image::cuda
