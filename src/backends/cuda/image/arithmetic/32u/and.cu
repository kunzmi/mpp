#if MPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(32u);
ForAllChannelsWithAlphaInvokeAndSrcC(32u);
ForAllChannelsWithAlphaInvokeAndSrcDevC(32u);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeAndInplaceC(32u);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
