#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(16s);
ForAllChannelsWithAlphaInvokeSetChannelDevC(16s);

} // namespace mpp::image::cuda
