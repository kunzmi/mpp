#if MPP_ENABLE_CUDA_BACKEND

#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(16s);
ForAllChannelsWithAlphaInvokeReplaceIfSrcC(16s);
ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(16s);

ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(16s);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(16s);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
