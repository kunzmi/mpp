#if OPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(64f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(64f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(64f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(64f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
