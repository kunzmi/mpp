#include "../rshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRShiftSrcC(16s);
ForAllChannelsWithAlphaInvokeRShiftInplaceC(16s);

} // namespace mpp::image::cuda
