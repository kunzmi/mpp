#if MPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(16u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(16u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(16u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(16u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
