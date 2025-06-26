#if MPP_ENABLE_CUDA_BACKEND

#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(8s);
ForAllChannelsWithAlphaInvokeSetChannelDevC(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
