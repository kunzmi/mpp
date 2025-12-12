#include "../rshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRShiftSrcC(32u);
ForAllChannelsWithAlphaInvokeRShiftInplaceC(32u);

} // namespace mpp::image::cuda
