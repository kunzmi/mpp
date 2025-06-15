#if OPP_ENABLE_CUDA_BACKEND

#include "../sub_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(16f);
ForAllChannelsWithAlphaInvokeSubSrcC(16f);
ForAllChannelsWithAlphaInvokeSubSrcDevC(16f);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeSubInplaceC(16f);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(16f);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(16f);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
