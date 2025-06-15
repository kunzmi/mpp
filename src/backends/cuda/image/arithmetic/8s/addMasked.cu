#if OPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(8s);
ForAllChannelsWithAlphaInvokeAddSrcSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeAddSrcCMask(8s);
ForAllChannelsWithAlphaInvokeAddSrcCScaleMask(8s);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(8s);
ForAllChannelsWithAlphaInvokeAddSrcDevCScaleMask(8s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(8s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(8s);
ForAllChannelsWithAlphaInvokeAddInplaceCScaleMask(8s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(8s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScaleMask(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
