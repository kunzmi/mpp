#if MPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrc(16f);
ForAllChannelsWithAlphaInvokeDivSrcC(16f);
ForAllChannelsWithAlphaInvokeDivSrcDevC(16f);
ForAllChannelsWithAlphaInvokeDivInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeDivInplaceC(16f);
ForAllChannelsWithAlphaInvokeDivInplaceDevC(16f);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeDivInvInplaceC(16f);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevC(16f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
