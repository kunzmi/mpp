#if OPP_ENABLE_CUDA_BACKEND

#include "../setChannel_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetChannelC(32fc);
ForAllChannelsNoAlphaInvokeSetChannelDevC(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
