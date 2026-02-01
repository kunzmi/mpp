#include "../sampling422Conversion_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSampling422ConversionC2P2Src(64f);
ForAllChannelsWithAlphaInvokeSampling422ConversionC2P3Src(64f);
ForAllChannelsWithAlphaInvokeSampling422ConversionP2C2Src(64f);
ForAllChannelsWithAlphaInvokeSampling422ConversionP3C2Src(64f);

} // namespace mpp::image::cuda
