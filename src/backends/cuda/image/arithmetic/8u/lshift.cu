#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(8u);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(8u);

} // namespace mpp::image::cuda
