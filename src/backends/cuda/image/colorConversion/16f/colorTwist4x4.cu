#include "../colorTwist4x4_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorTwist4x4Src(16f);
ForAllChannelsWithAlphaInvokeColorTwist4x4P4Src(16f);
ForAllChannelsWithAlphaInvokeColorTwist4x4P4SrcP4(16f);
ForAllChannelsWithAlphaInvokeColorTwist4x4SrcP4(16f);
ForAllChannelsWithAlphaInvokeColorTwist4x4Inplace(16f);

} // namespace mpp::image::cuda
