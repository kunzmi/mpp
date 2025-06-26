#if MPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(8u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(8u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(8u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(8u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
