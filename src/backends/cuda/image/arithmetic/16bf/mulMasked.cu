#if MPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(16bf);
ForAllChannelsWithAlphaInvokeMulSrcCMask(16bf);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
