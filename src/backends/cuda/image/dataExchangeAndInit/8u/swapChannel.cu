#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(8u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(8u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(8u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc2(8u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace2(8u);

} // namespace mpp::image::cuda
