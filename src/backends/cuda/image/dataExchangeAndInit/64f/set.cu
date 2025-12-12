#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(64f);
ForAllChannelsWithAlphaInvokeSetDevC(64f);

} // namespace mpp::image::cuda
