#if MPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(32s);
ForAllChannelsWithAlphaInvokeAndSrcC(32s);
ForAllChannelsWithAlphaInvokeAndSrcDevC(32s);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeAndInplaceC(32s);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
