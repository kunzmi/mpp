#include "../rshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRShiftSrcC(8s);
ForAllChannelsWithAlphaInvokeRShiftInplaceC(8s);

} // namespace mpp::image::cuda
