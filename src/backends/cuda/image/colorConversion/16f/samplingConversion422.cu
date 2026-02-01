#include "../sampling422Conversion_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSampling422ConversionC2P2Src(16f);
ForAllChannelsWithAlphaInvokeSampling422ConversionC2P3Src(16f);
ForAllChannelsWithAlphaInvokeSampling422ConversionP2C2Src(16f);
ForAllChannelsWithAlphaInvokeSampling422ConversionP3C2Src(16f);

} // namespace mpp::image::cuda
