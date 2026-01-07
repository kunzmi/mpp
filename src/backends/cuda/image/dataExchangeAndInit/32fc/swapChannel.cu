#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32fc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32fc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32fc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc2(32fc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace2(32fc);

} // namespace mpp::image::cuda
