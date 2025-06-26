#if MPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
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

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
