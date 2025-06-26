#if MPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(32f);
ForAllChannelsWithAlphaInvokeMulSrcCMask(32f);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(32f);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(32f);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(32f);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
