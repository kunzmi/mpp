#if MPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(8u);
ForAllChannelsWithAlphaInvokeSubSrcSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeSubSrcCMask(8u);
ForAllChannelsWithAlphaInvokeSubSrcCScaleMask(8u);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(8u);
ForAllChannelsWithAlphaInvokeSubSrcDevCScaleMask(8u);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(8u);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(8u);
ForAllChannelsWithAlphaInvokeSubInplaceCScaleMask(8u);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(8u);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScaleMask(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScaleMask(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(8u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScaleMask(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
