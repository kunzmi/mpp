#if MPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(32s);
ForAllChannelsWithAlphaInvokeMulSrcSrcScale(32s);
ForAllChannelsWithAlphaInvokeMulSrcC(32s);
ForAllChannelsWithAlphaInvokeMulSrcCScale(32s);
ForAllChannelsWithAlphaInvokeMulSrcDevC(32s);
ForAllChannelsWithAlphaInvokeMulSrcDevCScale(32s);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScale(32s);
ForAllChannelsWithAlphaInvokeMulInplaceC(32s);
ForAllChannelsWithAlphaInvokeMulInplaceCScale(32s);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(32s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScale(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
