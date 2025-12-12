#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(16bf);
ForAllChannelsWithAlphaInvokeSetDevCMask(16bf);

} // namespace mpp::image::cuda
