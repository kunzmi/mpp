#if MPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(64f);
ForAllChannelsWithAlphaInvokeMulSrcC(64f);
ForAllChannelsWithAlphaInvokeMulSrcDevC(64f);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeMulInplaceC(64f);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
