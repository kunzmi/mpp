#if MPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(16s);
ForAllChannelsWithAlphaInvokeAndSrcC(16s);
ForAllChannelsWithAlphaInvokeAndSrcDevC(16s);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeAndInplaceC(16s);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
