#if OPP_ENABLE_CUDA_BACKEND

#include "../setChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(8u);
ForAllChannelsWithAlphaInvokeSetChannelDevC(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
