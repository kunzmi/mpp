#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16u);

} // namespace mpp::image::cuda
