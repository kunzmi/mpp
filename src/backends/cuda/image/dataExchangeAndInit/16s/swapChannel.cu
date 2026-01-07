#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc2(16s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace2(16s);

} // namespace mpp::image::cuda
