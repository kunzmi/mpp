#if MPP_ENABLE_CUDA_BACKEND

#include "../or_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(16s);
ForAllChannelsWithAlphaInvokeOrSrcC(16s);
ForAllChannelsWithAlphaInvokeOrSrcDevC(16s);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeOrInplaceC(16s);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
