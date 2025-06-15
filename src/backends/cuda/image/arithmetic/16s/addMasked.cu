#if OPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcCMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcCScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcDevCScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceCScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScaleMask(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
