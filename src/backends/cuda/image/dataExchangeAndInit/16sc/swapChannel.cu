#if MPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(16sc);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(16sc);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
