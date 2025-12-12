#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(8u);
ForAllChannelsWithAlphaInvokeSetChannelDevC(8u);

} // namespace mpp::image::cuda
