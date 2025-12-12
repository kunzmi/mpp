#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(32s);
ForAllChannelsWithAlphaInvokeSetChannelDevC(32s);

} // namespace mpp::image::cuda
