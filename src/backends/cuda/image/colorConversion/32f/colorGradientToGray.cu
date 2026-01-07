#include "../colorGradientToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorGradientToGraySrc(32f);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP2(32f);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP3(32f);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP4(32f);

} // namespace mpp::image::cuda
