#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetChannelC(16sc);
ForAllChannelsNoAlphaInvokeSetChannelDevC(16sc);

} // namespace mpp::image::cuda
