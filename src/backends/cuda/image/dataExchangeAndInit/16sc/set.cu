#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetC(16sc);
ForAllChannelsNoAlphaInvokeSetDevC(16sc);

} // namespace mpp::image::cuda
