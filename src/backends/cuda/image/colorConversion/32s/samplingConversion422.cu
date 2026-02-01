#include "../sampling422Conversion_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSampling422ConversionC2P2Src(32s);
ForAllChannelsWithAlphaInvokeSampling422ConversionC2P3Src(32s);
ForAllChannelsWithAlphaInvokeSampling422ConversionP2C2Src(32s);
ForAllChannelsWithAlphaInvokeSampling422ConversionP3C2Src(32s);

} // namespace mpp::image::cuda
