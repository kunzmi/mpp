#if OPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
