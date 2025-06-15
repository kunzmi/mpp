#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
