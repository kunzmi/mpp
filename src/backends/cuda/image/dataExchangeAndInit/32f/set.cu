#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(32f);
ForAllChannelsWithAlphaInvokeSetDevC(32f);

} // namespace mpp::image::cuda
