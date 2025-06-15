#if OPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32sc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32sc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
