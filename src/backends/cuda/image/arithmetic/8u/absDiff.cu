#if OPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(8u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(8u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(8u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(8u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
