#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntRound(8s, 8u);
ForAllChannelsWithAlphaIntRound(8s, 16u);
ForAllChannelsWithAlphaIntRound(8s, 16s);
ForAllChannelsWithAlphaIntDiv(8s, 32u);
ForAllChannelsWithAlphaIntDiv(8s, 32s);
ForAllChannelsWithAlphaFloat(8s, 32f);
ForAllChannelsWithAlphaFloat(8s, 64f);
ForAllChannelsWithAlphaFloat(8s, 16f);
ForAllChannelsWithAlphaFloat(8s, 16bf);

} // namespace mpp::image::cuda
