#include "../colorTwist4x4_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorTwist4x4Src(8u);
ForAllChannelsWithAlphaInvokeColorTwist4x4P4Src(8u);
ForAllChannelsWithAlphaInvokeColorTwist4x4P4SrcP4(8u);
ForAllChannelsWithAlphaInvokeColorTwist4x4SrcP4(8u);
ForAllChannelsWithAlphaInvokeColorTwist4x4Inplace(8u);

} // namespace mpp::image::cuda
