#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(16f);
ForAllChannelsWithAlphaInvokeSetDevCMask(16f);

} // namespace mpp::image::cuda
