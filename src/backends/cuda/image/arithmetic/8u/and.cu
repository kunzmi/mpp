#if MPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(8u);
ForAllChannelsWithAlphaInvokeAndSrcC(8u);
ForAllChannelsWithAlphaInvokeAndSrcDevC(8u);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeAndInplaceC(8u);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
