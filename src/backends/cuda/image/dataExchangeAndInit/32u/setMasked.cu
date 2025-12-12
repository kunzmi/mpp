#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(32u);
ForAllChannelsWithAlphaInvokeSetDevCMask(32u);

} // namespace mpp::image::cuda
