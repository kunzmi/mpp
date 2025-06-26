#if MPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(32f);
ForAllChannelsWithAlphaInvokeMulSrcC(32f);
ForAllChannelsWithAlphaInvokeMulSrcDevC(32f);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeMulInplaceC(32f);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
