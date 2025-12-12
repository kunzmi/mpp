#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntRound(16f, 8s);
ForAllChannelsWithAlphaIntRound(16f, 8u);
ForAllChannelsWithAlphaIntRound(16f, 16s);
ForAllChannelsWithAlphaIntRound(16f, 16u);
ForAllChannelsWithAlphaIntRound(16f, 32u);
ForAllChannelsWithAlphaIntRound(16f, 32s);
ForAllChannelsWithAlphaFloat(16f, 16bf);
ForAllChannelsWithAlphaFloat(16f, 32f);
ForAllChannelsWithAlphaFloat(16f, 64f);

} // namespace mpp::image::cuda
