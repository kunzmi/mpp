#if MPP_ENABLE_CUDA_BACKEND

#include "../replaceIf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeReplaceIfSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeReplaceIfSrcC(32sc);
ForAllChannelsNoAlphaInvokeReplaceIfSrcDevC(32sc);

ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcC(32sc);
ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcDevC(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
