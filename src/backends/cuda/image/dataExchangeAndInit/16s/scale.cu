#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntRound(16s, 8s);
ForAllChannelsWithAlphaIntRound(16s, 8u);
ForAllChannelsWithAlphaIntRound(16s, 16u);
ForAllChannelsWithAlphaIntDiv(16s, 32u);
ForAllChannelsWithAlphaIntDiv(16s, 32s);
ForAllChannelsWithAlphaFloat(16s, 32f);
ForAllChannelsWithAlphaFloat(16s, 64f);
ForAllChannelsWithAlphaFloat(16s, 16f);
ForAllChannelsWithAlphaFloat(16s, 16bf);

} // namespace mpp::image::cuda
