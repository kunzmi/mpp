#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(16f);
ForAllChannelsWithAlphaInvokeSetChannelDevC(16f);

} // namespace mpp::image::cuda
