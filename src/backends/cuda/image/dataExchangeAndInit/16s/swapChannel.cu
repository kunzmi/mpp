#if MPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
