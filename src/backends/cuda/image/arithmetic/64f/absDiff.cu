#if MPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(64f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(64f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(64f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(64f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
