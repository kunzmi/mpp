#include "../set_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetC(32fc);
ForAllChannelsNoAlphaInvokeSetDevC(32fc);

} // namespace mpp::image::cuda
