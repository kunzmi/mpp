#if MPP_ENABLE_CUDA_BACKEND

#include "../sub_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(32u);
ForAllChannelsWithAlphaInvokeSubSrcSrcScale(32u);
ForAllChannelsWithAlphaInvokeSubSrcC(32u);
ForAllChannelsWithAlphaInvokeSubSrcCScale(32u);
ForAllChannelsWithAlphaInvokeSubSrcDevC(32u);
ForAllChannelsWithAlphaInvokeSubSrcDevCScale(32u);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScale(32u);
ForAllChannelsWithAlphaInvokeSubInplaceC(32u);
ForAllChannelsWithAlphaInvokeSubInplaceCScale(32u);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(32u);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScale(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScale(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScale(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScale(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
