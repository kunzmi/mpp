#include "../sampling422Conversion_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSampling422ConversionC2P2Src(8s);
ForAllChannelsWithAlphaInvokeSampling422ConversionC2P3Src(8s);
ForAllChannelsWithAlphaInvokeSampling422ConversionP2C2Src(8s);
ForAllChannelsWithAlphaInvokeSampling422ConversionP3C2Src(8s);

} // namespace mpp::image::cuda
