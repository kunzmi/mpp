#include "../colorGradientToGray_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorGradientToGraySrc(16s);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP2(16s);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP3(16s);
ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP4(16s);

} // namespace mpp::image::cuda
