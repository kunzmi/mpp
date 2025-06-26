#if MPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(8s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(8s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
