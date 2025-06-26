#if MPP_ENABLE_CUDA_BACKEND

#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(16u);
ForAllChannelsWithAlphaInvokeSetChannelDevC(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
