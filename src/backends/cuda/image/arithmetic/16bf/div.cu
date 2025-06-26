#if MPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeDivSrcC(16bf);
ForAllChannelsWithAlphaInvokeDivSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeDivInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeDivInplaceC(16bf);
ForAllChannelsWithAlphaInvokeDivInplaceDevC(16bf);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeDivInvInplaceC(16bf);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevC(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
