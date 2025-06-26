#if MPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMulSrcSrcMask(32fc);
ForAllChannelsNoAlphaInvokeMulSrcCMask(32fc);
ForAllChannelsNoAlphaInvokeMulSrcDevCMask(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceSrcMask(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceCMask(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceDevCMask(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
