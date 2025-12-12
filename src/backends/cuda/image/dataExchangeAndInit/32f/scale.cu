#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntRound(32f, 8s);
ForAllChannelsWithAlphaIntRound(32f, 8u);
ForAllChannelsWithAlphaIntRound(32f, 16s);
ForAllChannelsWithAlphaIntRound(32f, 16u);
ForAllChannelsWithAlphaIntRound(32f, 32u);
ForAllChannelsWithAlphaIntRound(32f, 32s);
ForAllChannelsWithAlphaFloat(32f, 64f);
ForAllChannelsWithAlphaFloat(32f, 16f);
ForAllChannelsWithAlphaFloat(32f, 16bf);

} // namespace mpp::image::cuda
