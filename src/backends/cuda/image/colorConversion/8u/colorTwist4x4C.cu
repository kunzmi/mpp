#include "../colorTwist4x4C_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorTwist4x4CSrc(8u);
ForAllChannelsWithAlphaInvokeColorTwist4x4CP4Src(8u);
ForAllChannelsWithAlphaInvokeColorTwist4x4CP4SrcP4(8u);
ForAllChannelsWithAlphaInvokeColorTwist4x4CSrcP4(8u);
ForAllChannelsWithAlphaInvokeColorTwist4x4CInplace(8u);

} // namespace mpp::image::cuda
