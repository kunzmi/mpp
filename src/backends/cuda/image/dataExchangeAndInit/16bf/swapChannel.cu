#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16bf);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16bf);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
