#if OPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(16f);
ForAllChannelsWithAlphaInvokeAddSrcCMask(16f);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(16f);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
