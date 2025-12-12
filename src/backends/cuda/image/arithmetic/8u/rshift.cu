#include "../rshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRShiftSrcC(8u);
ForAllChannelsWithAlphaInvokeRShiftInplaceC(8u);

} // namespace mpp::image::cuda
