#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(32u);
ForAllChannelsWithAlphaInvokeSetDevC(32u);

} // namespace mpp::image::cuda
