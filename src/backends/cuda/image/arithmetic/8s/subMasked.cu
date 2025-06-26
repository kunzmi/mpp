#if MPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(8s);
ForAllChannelsWithAlphaInvokeSubSrcSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeSubSrcCMask(8s);
ForAllChannelsWithAlphaInvokeSubSrcCScaleMask(8s);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(8s);
ForAllChannelsWithAlphaInvokeSubSrcDevCScaleMask(8s);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(8s);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(8s);
ForAllChannelsWithAlphaInvokeSubInplaceCScaleMask(8s);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(8s);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScaleMask(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScaleMask(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScaleMask(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
