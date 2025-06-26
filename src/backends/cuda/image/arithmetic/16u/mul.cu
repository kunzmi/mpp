#if MPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(16u);
ForAllChannelsWithAlphaInvokeMulSrcSrcScale(16u);
ForAllChannelsWithAlphaInvokeMulSrcC(16u);
ForAllChannelsWithAlphaInvokeMulSrcCScale(16u);
ForAllChannelsWithAlphaInvokeMulSrcDevC(16u);
ForAllChannelsWithAlphaInvokeMulSrcDevCScale(16u);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScale(16u);
ForAllChannelsWithAlphaInvokeMulInplaceC(16u);
ForAllChannelsWithAlphaInvokeMulInplaceCScale(16u);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(16u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScale(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
