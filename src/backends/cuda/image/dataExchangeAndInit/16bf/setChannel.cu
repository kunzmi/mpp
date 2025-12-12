#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetChannelC(16bf);
ForAllChannelsWithAlphaInvokeSetChannelDevC(16bf);

} // namespace mpp::image::cuda
