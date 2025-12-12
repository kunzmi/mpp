#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(32s);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(32s);

} // namespace mpp::image::cuda
