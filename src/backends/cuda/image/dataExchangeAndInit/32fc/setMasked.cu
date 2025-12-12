#include "../setMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetCMask(32fc);
ForAllChannelsNoAlphaInvokeSetDevCMask(32fc);

} // namespace mpp::image::cuda
