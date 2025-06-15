#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
