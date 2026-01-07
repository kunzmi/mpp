#include "../colorToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorToGraySrc(32f);
ForAllChannelsWithAlphaInvokeColorToGraySrcP2(32f);
ForAllChannelsWithAlphaInvokeColorToGraySrcP3(32f);
ForAllChannelsWithAlphaInvokeColorToGraySrcP4(32f);

} // namespace mpp::image::cuda
