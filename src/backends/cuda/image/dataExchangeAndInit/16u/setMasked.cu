#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(16u);
ForAllChannelsWithAlphaInvokeSetDevCMask(16u);

} // namespace mpp::image::cuda
