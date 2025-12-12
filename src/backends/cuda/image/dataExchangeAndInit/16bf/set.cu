#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetC(16bf);
ForAllChannelsWithAlphaInvokeSetDevC(16bf);

} // namespace mpp::image::cuda
