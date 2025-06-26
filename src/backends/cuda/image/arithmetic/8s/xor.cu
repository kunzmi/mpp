#if MPP_ENABLE_CUDA_BACKEND

#include "../xor_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(8s);
ForAllChannelsWithAlphaInvokeXorSrcC(8s);
ForAllChannelsWithAlphaInvokeXorSrcDevC(8s);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeXorInplaceC(8s);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
