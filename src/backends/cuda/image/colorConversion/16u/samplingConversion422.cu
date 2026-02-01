#include "../sampling422Conversion_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSampling422ConversionC2P2Src(16u);
ForAllChannelsWithAlphaInvokeSampling422ConversionC2P3Src(16u);
ForAllChannelsWithAlphaInvokeSampling422ConversionP2C2Src(16u);
ForAllChannelsWithAlphaInvokeSampling422ConversionP3C2Src(16u);

} // namespace mpp::image::cuda
