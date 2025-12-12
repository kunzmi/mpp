#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(32f);
ForAllChannelsWithAlphaInvokeSetChannelDevC(32f);

} // namespace mpp::image::cuda
