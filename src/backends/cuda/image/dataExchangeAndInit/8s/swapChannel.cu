#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(8s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(8s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(8s);

} // namespace mpp::image::cuda
