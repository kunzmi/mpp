#if OPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(32f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(32f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(32f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(32f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
