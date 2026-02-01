#include "../sampling422Conversion_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSampling422ConversionC2P2Src(16bf);
ForAllChannelsWithAlphaInvokeSampling422ConversionC2P3Src(16bf);
ForAllChannelsWithAlphaInvokeSampling422ConversionP2C2Src(16bf);
ForAllChannelsWithAlphaInvokeSampling422ConversionP3C2Src(16bf);

} // namespace mpp::image::cuda
