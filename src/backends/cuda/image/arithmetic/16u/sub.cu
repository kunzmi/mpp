#if MPP_ENABLE_CUDA_BACKEND

#include "../sub_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(16u);
ForAllChannelsWithAlphaInvokeSubSrcSrcScale(16u);
ForAllChannelsWithAlphaInvokeSubSrcC(16u);
ForAllChannelsWithAlphaInvokeSubSrcCScale(16u);
ForAllChannelsWithAlphaInvokeSubSrcDevC(16u);
ForAllChannelsWithAlphaInvokeSubSrcDevCScale(16u);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScale(16u);
ForAllChannelsWithAlphaInvokeSubInplaceC(16u);
ForAllChannelsWithAlphaInvokeSubInplaceCScale(16u);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(16u);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScale(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScale(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScale(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScale(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
