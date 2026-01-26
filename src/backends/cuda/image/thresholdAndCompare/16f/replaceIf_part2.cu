#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

// ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(16f);
ForAllChannelsWithAlphaInvokeReplaceIfSrcC(16f);
// ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(16f);
// ForAllChannelsWithAlphaInvokeReplaceIfSrc(16f);

// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(16f);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(16f);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(16f);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrc(16f);

} // namespace mpp::image::cuda
