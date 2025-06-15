#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
