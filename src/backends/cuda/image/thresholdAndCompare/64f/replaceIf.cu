#if MPP_ENABLE_CUDA_BACKEND

#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(64f);
ForAllChannelsWithAlphaInvokeReplaceIfSrcC(64f);
ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(64f);
ForAllChannelsWithAlphaInvokeReplaceIfSrc(64f);

ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(64f);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(64f);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(64f);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrc(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
