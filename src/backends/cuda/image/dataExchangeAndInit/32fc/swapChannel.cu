#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32fc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32fc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
