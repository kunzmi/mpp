#if MPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32f);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32f);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
