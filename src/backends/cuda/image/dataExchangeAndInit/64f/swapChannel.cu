#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(64f);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(64f);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
