#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(8u);
ForAllChannelsWithAlphaInvokeSetDevCMask(8u);

} // namespace mpp::image::cuda
