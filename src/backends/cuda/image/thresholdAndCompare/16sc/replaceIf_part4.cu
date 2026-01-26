#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

// ForAllChannelsNoAlphaInvokeReplaceIfSrcSrc(16sc);
// ForAllChannelsNoAlphaInvokeReplaceIfSrcC(16sc);
// ForAllChannelsNoAlphaInvokeReplaceIfSrcDevC(16sc);

ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcSrc(16sc);
// ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcC(16sc);
// ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcDevC(16sc);

} // namespace mpp::image::cuda
