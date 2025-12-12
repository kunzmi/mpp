#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(32s);
ForAllChannelsWithAlphaInvokeSetDevC(32s);

} // namespace mpp::image::cuda
