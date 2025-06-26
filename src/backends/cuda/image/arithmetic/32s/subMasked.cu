#if MPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(32s);
ForAllChannelsWithAlphaInvokeSubSrcSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeSubSrcCMask(32s);
ForAllChannelsWithAlphaInvokeSubSrcCScaleMask(32s);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(32s);
ForAllChannelsWithAlphaInvokeSubSrcDevCScaleMask(32s);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(32s);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(32s);
ForAllChannelsWithAlphaInvokeSubInplaceCScaleMask(32s);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(32s);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScaleMask(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScaleMask(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScaleMask(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
