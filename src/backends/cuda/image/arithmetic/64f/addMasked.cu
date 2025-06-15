#if OPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(64f);
ForAllChannelsWithAlphaInvokeAddSrcCMask(64f);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(64f);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(64f);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(64f);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
