#if OPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(16u);
ForAllChannelsWithAlphaInvokeSubSrcSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeSubSrcCMask(16u);
ForAllChannelsWithAlphaInvokeSubSrcCScaleMask(16u);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(16u);
ForAllChannelsWithAlphaInvokeSubSrcDevCScaleMask(16u);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(16u);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(16u);
ForAllChannelsWithAlphaInvokeSubInplaceCScaleMask(16u);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(16u);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScaleMask(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScaleMask(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(16u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScaleMask(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
