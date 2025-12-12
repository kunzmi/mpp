#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32f);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32f);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32f);

} // namespace mpp::image::cuda
