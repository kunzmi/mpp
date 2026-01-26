#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

// ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(8u);
// ForAllChannelsWithAlphaInvokeReplaceIfSrcC(8u);
// ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(8u);

// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(8u);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(8u);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(8u);

} // namespace mpp::image::cuda
