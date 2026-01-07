#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16sc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16sc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16sc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc2(16sc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace2(16sc);

} // namespace mpp::image::cuda
