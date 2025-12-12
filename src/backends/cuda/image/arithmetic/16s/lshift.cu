#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(16s);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(16s);

} // namespace mpp::image::cuda
