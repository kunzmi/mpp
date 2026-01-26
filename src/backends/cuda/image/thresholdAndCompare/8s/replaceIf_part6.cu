#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

// ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(8s);
// ForAllChannelsWithAlphaInvokeReplaceIfSrcC(8s);
// ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(8s);

// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(8s);
// ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(8s);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(8s);

} // namespace mpp::image::cuda
