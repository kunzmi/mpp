#if MPP_ENABLE_CUDA_BACKEND

#include "../xor_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(8u);
ForAllChannelsWithAlphaInvokeXorSrcC(8u);
ForAllChannelsWithAlphaInvokeXorSrcDevC(8u);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeXorInplaceC(8u);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
