#include "../colorTwist4x4C_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorTwist4x4CSrc(32f);
ForAllChannelsWithAlphaInvokeColorTwist4x4CP4Src(32f);
ForAllChannelsWithAlphaInvokeColorTwist4x4CP4SrcP4(32f);
ForAllChannelsWithAlphaInvokeColorTwist4x4CSrcP4(32f);
ForAllChannelsWithAlphaInvokeColorTwist4x4CInplace(32f);

} // namespace mpp::image::cuda
