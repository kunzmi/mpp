#if OPP_ENABLE_CUDA_BACKEND

#include "../setChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(16bf);
ForAllChannelsWithAlphaInvokeSetChannelDevC(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
