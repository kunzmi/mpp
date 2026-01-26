#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(32f);
// ForAllChannelsWithAlphaInvokeReplaceIfSrcC(32f);
// ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(32f);
// ForAllChannelsWithAlphaInvokeReplaceIfSrc(32f);

// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(32f);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(32f);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(32f);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrc(32f);

} // namespace mpp::image::cuda
