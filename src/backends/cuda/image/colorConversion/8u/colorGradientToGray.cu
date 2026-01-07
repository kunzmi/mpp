#include "../colorGradientToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorGradientToGraySrc(8u);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP2(8u);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP3(8u);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP4(8u);

} // namespace mpp::image::cuda
