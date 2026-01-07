#include "../colorGradientToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorGradientToGraySrc(16u);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP2(16u);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP3(16u);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP4(16u);

} // namespace mpp::image::cuda
