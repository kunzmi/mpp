#if MPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(16bf);
ForAllChannelsWithAlphaInvokeAddSrcCMask(16bf);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
