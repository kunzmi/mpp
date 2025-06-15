#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(8u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(8u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
