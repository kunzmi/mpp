#if MPP_ENABLE_CUDA_BACKEND

#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(16u);
ForAllChannelsWithAlphaInvokeReplaceIfSrcC(16u);
ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(16u);

ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(16u);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(16u);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
