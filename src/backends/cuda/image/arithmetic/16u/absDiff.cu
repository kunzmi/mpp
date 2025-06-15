#if OPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(16u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(16u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(16u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(16u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
