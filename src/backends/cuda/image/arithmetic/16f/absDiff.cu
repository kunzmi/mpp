#if OPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(16f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(16f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(16f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(16f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
