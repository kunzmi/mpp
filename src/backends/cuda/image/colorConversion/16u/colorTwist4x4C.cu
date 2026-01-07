#include "../colorTwist4x4C_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorTwist4x4CSrc(16u);
ForAllChannelsWithAlphaInvokeColorTwist4x4CP4Src(16u);
ForAllChannelsWithAlphaInvokeColorTwist4x4CP4SrcP4(16u);
ForAllChannelsWithAlphaInvokeColorTwist4x4CSrcP4(16u);
ForAllChannelsWithAlphaInvokeColorTwist4x4CInplace(16u);

} // namespace mpp::image::cuda
