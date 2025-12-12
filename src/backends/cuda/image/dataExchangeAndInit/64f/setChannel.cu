#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(64f);
ForAllChannelsWithAlphaInvokeSetChannelDevC(64f);

} // namespace mpp::image::cuda
