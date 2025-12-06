#if MPP_ENABLE_CUDA_BACKEND

#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeReplaceIfSrcC(16bf);
ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeReplaceIfSrc(16bf);

ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(16bf);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrc(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
