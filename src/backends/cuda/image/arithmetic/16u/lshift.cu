#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(16u);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(16u);

} // namespace mpp::image::cuda
