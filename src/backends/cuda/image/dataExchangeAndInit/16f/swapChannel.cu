#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16f);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16f);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16f);

} // namespace mpp::image::cuda
