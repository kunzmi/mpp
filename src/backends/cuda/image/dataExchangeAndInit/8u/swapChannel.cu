#if MPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(8u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(8u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
