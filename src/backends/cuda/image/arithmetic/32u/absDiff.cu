#if MPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(32u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(32u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(32u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(32u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
