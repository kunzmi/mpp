#if MPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMulSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeMulSrcC(32fc);
ForAllChannelsNoAlphaInvokeMulSrcDevC(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceC(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceDevC(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
