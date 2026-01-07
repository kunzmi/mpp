#include "../colorToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorToGraySrc(16f);
ForAllChannelsWithAlphaInvokeColorToGraySrcP2(16f);
ForAllChannelsWithAlphaInvokeColorToGraySrcP3(16f);
ForAllChannelsWithAlphaInvokeColorToGraySrcP4(16f);

} // namespace mpp::image::cuda
