#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

// ForAllChannelsNoAlphaInvokeReplaceIfSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeReplaceIfSrcC(32fc);
// ForAllChannelsNoAlphaInvokeReplaceIfSrcDevC(32fc);
// ForAllChannelsNoAlphaInvokeReplaceIfSrc(32fc);

// ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcSrc(32fc);
// ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcC(32fc);
// ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcDevC(32fc);
// ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrc(32fc);

} // namespace mpp::image::cuda
