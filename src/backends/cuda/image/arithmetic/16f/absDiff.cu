#if MPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(16f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(16f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(16f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(16f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(16f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
