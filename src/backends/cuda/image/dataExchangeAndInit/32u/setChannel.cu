#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(32u);
ForAllChannelsWithAlphaInvokeSetChannelDevC(32u);

} // namespace mpp::image::cuda
