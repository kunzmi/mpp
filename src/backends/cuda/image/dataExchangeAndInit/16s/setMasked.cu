#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSetCMask(16s);
ForAllChannelsWithAlphaInvokeSetDevCMask(16s);

} // namespace mpp::image::cuda
