#if OPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(64f);
ForAllChannelsWithAlphaInvokeSubSrcCMask(64f);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(64f);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(64f);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(64f);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(64f);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(64f);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(64f);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
