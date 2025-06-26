#if MPP_ENABLE_CUDA_BACKEND

#include "../swapChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSwapChannelSrc(32s);
ForAllChannelsNoAlphaInvokeSwapChannelSrc34(32s);
ForAllChannelsNoAlphaInvokeSwapChannelInplace(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
