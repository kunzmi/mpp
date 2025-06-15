#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(8s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(8s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
