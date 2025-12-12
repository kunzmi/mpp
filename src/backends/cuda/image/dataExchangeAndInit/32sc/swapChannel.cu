#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32sc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32sc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32sc);

} // namespace mpp::image::cuda
