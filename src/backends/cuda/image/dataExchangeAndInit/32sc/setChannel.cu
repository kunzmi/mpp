#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetChannelC(32sc);
ForAllChannelsNoAlphaInvokeSetChannelDevC(32sc);

} // namespace mpp::image::cuda
