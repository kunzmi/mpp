#if OPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
