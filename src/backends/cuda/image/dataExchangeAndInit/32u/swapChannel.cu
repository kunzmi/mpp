#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc2(32u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace2(32u);

} // namespace mpp::image::cuda
