#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(64f);
ForAllChannelsWithAlphaInvokeSetDevCMask(64f);

} // namespace mpp::image::cuda
