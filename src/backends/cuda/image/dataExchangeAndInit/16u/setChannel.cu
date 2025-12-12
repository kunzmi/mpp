#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(16u);
ForAllChannelsWithAlphaInvokeSetChannelDevC(16u);

} // namespace mpp::image::cuda
