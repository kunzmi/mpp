#include "../colorTwist4x4_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorTwist4x4Src(16s);
ForAllChannelsWithAlphaInvokeColorTwist4x4P4Src(16s);
ForAllChannelsWithAlphaInvokeColorTwist4x4P4SrcP4(16s);
ForAllChannelsWithAlphaInvokeColorTwist4x4SrcP4(16s);
ForAllChannelsWithAlphaInvokeColorTwist4x4Inplace(16s);

} // namespace mpp::image::cuda
