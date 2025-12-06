#if MPP_ENABLE_CUDA_BACKEND

#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(32u);
ForAllChannelsWithAlphaInvokeReplaceIfSrcC(32u);
ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(32u);

ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(32u);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(32u);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
