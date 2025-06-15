#if OPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcCMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcCScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcDevCScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceCScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScaleMask(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
