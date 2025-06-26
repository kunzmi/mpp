#if MPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(16u);
ForAllChannelsWithAlphaInvokeDivSrcCScale(16u);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(16u);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(16u);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(16u);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(16u);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(16u);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(16u);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
