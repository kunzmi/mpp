#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(32u);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(32u);

} // namespace mpp::image::cuda
