#include "../lshift_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(8s);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(8s);

} // namespace mpp::image::cuda
