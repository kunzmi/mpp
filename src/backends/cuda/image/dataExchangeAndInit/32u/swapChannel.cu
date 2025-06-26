#if MPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32u);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32u);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
