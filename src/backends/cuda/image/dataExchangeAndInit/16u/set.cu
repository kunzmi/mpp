#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(16u);
ForAllChannelsWithAlphaInvokeSetDevC(16u);

} // namespace mpp::image::cuda
