#if MPP_ENABLE_CUDA_BACKEND

#include "../or_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(8u);
ForAllChannelsWithAlphaInvokeOrSrcC(8u);
ForAllChannelsWithAlphaInvokeOrSrcDevC(8u);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeOrInplaceC(8u);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
