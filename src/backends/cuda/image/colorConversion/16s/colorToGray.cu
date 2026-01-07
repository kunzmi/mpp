#include "../colorToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorToGraySrc(16s);
ForAllChannelsWithAlphaInvokeColorToGraySrcP2(16s);
ForAllChannelsWithAlphaInvokeColorToGraySrcP3(16s);
ForAllChannelsWithAlphaInvokeColorToGraySrcP4(16s);

} // namespace mpp::image::cuda
