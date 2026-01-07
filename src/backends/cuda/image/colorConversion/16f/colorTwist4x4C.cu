#include "../colorTwist4x4C_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorTwist4x4CSrc(16f);
ForAllChannelsWithAlphaInvokeColorTwist4x4CP4Src(16f);
ForAllChannelsWithAlphaInvokeColorTwist4x4CP4SrcP4(16f);
ForAllChannelsWithAlphaInvokeColorTwist4x4CSrcP4(16f);
ForAllChannelsWithAlphaInvokeColorTwist4x4CInplace(16f);

} // namespace mpp::image::cuda
