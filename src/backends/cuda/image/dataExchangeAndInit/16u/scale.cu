#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntRound(16u, 8s);
ForAllChannelsWithAlphaIntRound(16u, 8u);
ForAllChannelsWithAlphaIntRound(16u, 16s);
ForAllChannelsWithAlphaIntDiv(16u, 32u);
ForAllChannelsWithAlphaIntDiv(16u, 32s);
ForAllChannelsWithAlphaFloat(16u, 32f);
ForAllChannelsWithAlphaFloat(16u, 64f);
ForAllChannelsWithAlphaFloat(16u, 16f);
ForAllChannelsWithAlphaFloat(16u, 16bf);

} // namespace mpp::image::cuda
