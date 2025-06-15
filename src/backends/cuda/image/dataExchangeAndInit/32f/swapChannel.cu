#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32f);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32f);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
