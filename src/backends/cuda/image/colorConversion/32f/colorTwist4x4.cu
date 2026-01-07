#include "../colorTwist4x4_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorTwist4x4Src(32f);
ForAllChannelsWithAlphaInvokeColorTwist4x4P4Src(32f);
ForAllChannelsWithAlphaInvokeColorTwist4x4P4SrcP4(32f);
ForAllChannelsWithAlphaInvokeColorTwist4x4SrcP4(32f);
ForAllChannelsWithAlphaInvokeColorTwist4x4Inplace(32f);

} // namespace mpp::image::cuda
