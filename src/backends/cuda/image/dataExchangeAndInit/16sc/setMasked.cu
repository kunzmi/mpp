#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetCMask(16sc);
ForAllChannelsNoAlphaInvokeSetDevCMask(16sc);

} // namespace mpp::image::cuda
