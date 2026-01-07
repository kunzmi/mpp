#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc2(32s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace2(32s);

} // namespace mpp::image::cuda
