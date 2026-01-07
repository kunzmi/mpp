#include "../colorToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorToGraySrc(16u);
ForAllChannelsWithAlphaInvokeColorToGraySrcP2(16u);
ForAllChannelsWithAlphaInvokeColorToGraySrcP3(16u);
ForAllChannelsWithAlphaInvokeColorToGraySrcP4(16u);

} // namespace mpp::image::cuda
