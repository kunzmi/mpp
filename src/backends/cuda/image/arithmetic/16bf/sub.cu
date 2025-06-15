#if OPP_ENABLE_CUDA_BACKEND

#include "../sub_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeSubSrcC(16bf);
ForAllChannelsWithAlphaInvokeSubSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeSubInplaceC(16bf);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(16bf);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(16bf);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
