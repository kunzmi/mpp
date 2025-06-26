#if MPP_ENABLE_CUDA_BACKEND

#include "../sub_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
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

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
