#if MPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(16f);
ForAllChannelsWithAlphaInvokeMulSrcC(16f);
ForAllChannelsWithAlphaInvokeMulSrcDevC(16f);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeMulInplaceC(16f);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(16f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
