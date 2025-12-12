#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(32f);
ForAllChannelsWithAlphaInvokeSetDevCMask(32f);

} // namespace mpp::image::cuda
