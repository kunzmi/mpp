#if OPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(32u);
ForAllChannelsWithAlphaInvokeSubSrcSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeSubSrcCMask(32u);
ForAllChannelsWithAlphaInvokeSubSrcCScaleMask(32u);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(32u);
ForAllChannelsWithAlphaInvokeSubSrcDevCScaleMask(32u);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(32u);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(32u);
ForAllChannelsWithAlphaInvokeSubInplaceCScaleMask(32u);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(32u);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScaleMask(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScaleMask(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(32u);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScaleMask(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
