#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
