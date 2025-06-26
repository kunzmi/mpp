#if MPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddSrcSrcMask(16sc);
ForAllChannelsNoAlphaInvokeAddSrcSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeAddSrcCMask(16sc);
ForAllChannelsNoAlphaInvokeAddSrcCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeAddSrcDevCMask(16sc);
ForAllChannelsNoAlphaInvokeAddSrcDevCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceSrcMask(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceCMask(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceDevCMask(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceDevCScaleMask(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
