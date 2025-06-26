#if MPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeMulSrcC(16bf);
ForAllChannelsWithAlphaInvokeMulSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceC(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
