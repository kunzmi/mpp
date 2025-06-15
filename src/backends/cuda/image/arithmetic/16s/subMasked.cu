#if OPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(16s);
ForAllChannelsWithAlphaInvokeSubSrcSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeSubSrcCMask(16s);
ForAllChannelsWithAlphaInvokeSubSrcCScaleMask(16s);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(16s);
ForAllChannelsWithAlphaInvokeSubSrcDevCScaleMask(16s);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(16s);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(16s);
ForAllChannelsWithAlphaInvokeSubInplaceCScaleMask(16s);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(16s);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScaleMask(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScaleMask(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScaleMask(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
