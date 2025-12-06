#if MPP_ENABLE_CUDA_BACKEND

#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(32s);
ForAllChannelsWithAlphaInvokeReplaceIfSrcC(32s);
ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(32s);

ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(32s);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(32s);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
