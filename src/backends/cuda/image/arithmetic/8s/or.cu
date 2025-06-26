#if MPP_ENABLE_CUDA_BACKEND

#include "../or_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(8s);
ForAllChannelsWithAlphaInvokeOrSrcC(8s);
ForAllChannelsWithAlphaInvokeOrSrcDevC(8s);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeOrInplaceC(8s);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
