#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntDiv(32u, 8s);
ForAllChannelsWithAlphaIntDiv(32u, 8u);
ForAllChannelsWithAlphaIntDiv(32u, 16s);
ForAllChannelsWithAlphaIntDiv(32u, 16u);
ForAllChannelsWithAlphaIntDiv(32u, 32s);
ForAllChannelsWithAlphaFloat(32u, 32f);
ForAllChannelsWithAlphaFloat(32u, 64f);
ForAllChannelsWithAlphaFloat(32u, 16f);
ForAllChannelsWithAlphaFloat(32u, 16bf);

} // namespace mpp::image::cuda
