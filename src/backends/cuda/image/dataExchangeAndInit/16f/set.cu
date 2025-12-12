#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(16f);
ForAllChannelsWithAlphaInvokeSetDevC(16f);

} // namespace mpp::image::cuda
