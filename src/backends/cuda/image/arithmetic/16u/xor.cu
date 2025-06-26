#if MPP_ENABLE_CUDA_BACKEND

#include "../xor_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(16u);
ForAllChannelsWithAlphaInvokeXorSrcC(16u);
ForAllChannelsWithAlphaInvokeXorSrcDevC(16u);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeXorInplaceC(16u);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
