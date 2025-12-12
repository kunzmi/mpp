#include "../rshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRShiftSrcC(32s);
ForAllChannelsWithAlphaInvokeRShiftInplaceC(32s);

} // namespace mpp::image::cuda
