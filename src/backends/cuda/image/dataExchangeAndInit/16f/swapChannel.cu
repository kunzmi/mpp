#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16f);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16f);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
