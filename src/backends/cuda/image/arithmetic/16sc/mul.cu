#if MPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMulSrcSrc(16sc);
ForAllChannelsNoAlphaInvokeMulSrcSrcScale(16sc);
ForAllChannelsNoAlphaInvokeMulSrcC(16sc);
ForAllChannelsNoAlphaInvokeMulSrcCScale(16sc);
ForAllChannelsNoAlphaInvokeMulSrcDevC(16sc);
ForAllChannelsNoAlphaInvokeMulSrcDevCScale(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceSrc(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceSrcScale(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceC(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceCScale(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceDevC(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceDevCScale(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
