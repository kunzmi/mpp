#if MPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32fc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32fc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
