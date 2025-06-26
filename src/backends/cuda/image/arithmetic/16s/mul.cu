#if MPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(16s);
ForAllChannelsWithAlphaInvokeMulSrcSrcScale(16s);
ForAllChannelsWithAlphaInvokeMulSrcC(16s);
ForAllChannelsWithAlphaInvokeMulSrcCScale(16s);
ForAllChannelsWithAlphaInvokeMulSrcDevC(16s);
ForAllChannelsWithAlphaInvokeMulSrcDevCScale(16s);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScale(16s);
ForAllChannelsWithAlphaInvokeMulInplaceC(16s);
ForAllChannelsWithAlphaInvokeMulInplaceCScale(16s);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(16s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScale(16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
