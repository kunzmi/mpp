#if OPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(16u);
ForAllChannelsWithAlphaInvokeAddSrcSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeAddSrcCMask(16u);
ForAllChannelsWithAlphaInvokeAddSrcCScaleMask(16u);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(16u);
ForAllChannelsWithAlphaInvokeAddSrcDevCScaleMask(16u);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(16u);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(16u);
ForAllChannelsWithAlphaInvokeAddInplaceCScaleMask(16u);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(16u);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScaleMask(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
