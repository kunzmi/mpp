#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(64f);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(64f);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(64f);

} // namespace mpp::image::cuda
