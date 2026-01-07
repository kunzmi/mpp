#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(8s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(8s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(8s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc2(8s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace2(8s);

} // namespace mpp::image::cuda
