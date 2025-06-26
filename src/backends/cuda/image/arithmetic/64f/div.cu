#if MPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrc(64f);
ForAllChannelsWithAlphaInvokeDivSrcC(64f);
ForAllChannelsWithAlphaInvokeDivSrcDevC(64f);
ForAllChannelsWithAlphaInvokeDivInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeDivInplaceC(64f);
ForAllChannelsWithAlphaInvokeDivInplaceDevC(64f);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeDivInvInplaceC(64f);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevC(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
