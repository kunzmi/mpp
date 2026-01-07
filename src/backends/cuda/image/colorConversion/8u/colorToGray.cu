#include "../colorToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorToGraySrc(8u);
ForAllChannelsWithAlphaInvokeColorToGraySrcP2(8u);
ForAllChannelsWithAlphaInvokeColorToGraySrcP3(8u);
ForAllChannelsWithAlphaInvokeColorToGraySrcP4(8u);

} // namespace mpp::image::cuda
