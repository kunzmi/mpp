#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

// ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(16u);
// ForAllChannelsWithAlphaInvokeReplaceIfSrcC(16u);
// ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(16u);

ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(16u);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(16u);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(16u);

} // namespace mpp::image::cuda
