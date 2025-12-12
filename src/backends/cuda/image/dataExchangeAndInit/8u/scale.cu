#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntRound(8u, 8s);
ForAllChannelsWithAlphaIntRound(8u, 16u);
ForAllChannelsWithAlphaIntRound(8u, 16s);
ForAllChannelsWithAlphaIntDiv(8u, 32u);
ForAllChannelsWithAlphaIntDiv(8u, 32s);
ForAllChannelsWithAlphaFloat(8u, 32f);
ForAllChannelsWithAlphaFloat(8u, 64f);
ForAllChannelsWithAlphaFloat(8u, 16f);
ForAllChannelsWithAlphaFloat(8u, 16bf);

} // namespace mpp::image::cuda
