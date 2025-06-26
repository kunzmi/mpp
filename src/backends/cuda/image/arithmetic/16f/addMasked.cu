#if MPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(16f);
ForAllChannelsWithAlphaInvokeAddSrcCMask(16f);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(16f);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(16f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
