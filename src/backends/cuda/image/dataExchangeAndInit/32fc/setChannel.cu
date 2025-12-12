#include "../setChannel_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetChannelC(32fc);
ForAllChannelsNoAlphaInvokeSetChannelDevC(32fc);

} // namespace mpp::image::cuda
