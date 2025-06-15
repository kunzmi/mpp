#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16sc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16sc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
