#if OPP_ENABLE_CUDA_BACKEND

#include "../sub_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(8u);
ForAllChannelsWithAlphaInvokeSubSrcSrcScale(8u);
ForAllChannelsWithAlphaInvokeSubSrcC(8u);
ForAllChannelsWithAlphaInvokeSubSrcCScale(8u);
ForAllChannelsWithAlphaInvokeSubSrcDevC(8u);
ForAllChannelsWithAlphaInvokeSubSrcDevCScale(8u);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScale(8u);
ForAllChannelsWithAlphaInvokeSubInplaceC(8u);
ForAllChannelsWithAlphaInvokeSubInplaceCScale(8u);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(8u);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScale(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScale(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScale(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScale(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
