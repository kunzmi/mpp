#if OPP_ENABLE_CUDA_BACKEND

#include "../absDiff_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(32u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(32u);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(32u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(32u);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
