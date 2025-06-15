#if OPP_ENABLE_CUDA_BACKEND

#include "../subMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(16bf);
ForAllChannelsWithAlphaInvokeSubSrcCMask(16bf);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(16bf);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(16bf);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
