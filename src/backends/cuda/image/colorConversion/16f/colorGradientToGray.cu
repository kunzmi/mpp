#include "../colorGradientToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorGradientToGraySrc(16f);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP2(16f);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP3(16f);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP4(16f);

} // namespace mpp::image::cuda
