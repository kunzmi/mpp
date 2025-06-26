#if MPP_ENABLE_CUDA_BACKEND

#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetChannelC(32fc);
ForAllChannelsNoAlphaInvokeSetChannelDevC(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
