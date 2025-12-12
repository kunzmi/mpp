#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntRound(64f, 8s);
ForAllChannelsWithAlphaIntRound(64f, 8u);
ForAllChannelsWithAlphaIntRound(64f, 16s);
ForAllChannelsWithAlphaIntRound(64f, 16u);
ForAllChannelsWithAlphaIntRound(64f, 32u);
ForAllChannelsWithAlphaIntRound(64f, 32s);
ForAllChannelsWithAlphaFloat(64f, 32f);
ForAllChannelsWithAlphaFloat(64f, 16f);
ForAllChannelsWithAlphaFloat(64f, 16bf);

} // namespace mpp::image::cuda
