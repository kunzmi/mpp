#if OPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(16f);
ForAllChannelsWithAlphaInvokeSubSrcCMask(16f);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(16f);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(16f);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
