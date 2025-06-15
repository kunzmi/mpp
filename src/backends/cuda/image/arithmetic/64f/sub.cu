#if OPP_ENABLE_CUDA_BACKEND

#include "../sub_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(64f);
ForAllChannelsWithAlphaInvokeSubSrcC(64f);
ForAllChannelsWithAlphaInvokeSubSrcDevC(64f);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeSubInplaceC(64f);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(64f);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(64f);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
