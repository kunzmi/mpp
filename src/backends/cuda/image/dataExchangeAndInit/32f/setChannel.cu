#if OPP_ENABLE_CUDA_BACKEND

#include "../setChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(32f);
ForAllChannelsWithAlphaInvokeSetChannelDevC(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
