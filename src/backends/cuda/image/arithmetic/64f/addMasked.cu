#if MPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(64f);
ForAllChannelsWithAlphaInvokeAddSrcCMask(64f);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(64f);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(64f);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(64f);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
