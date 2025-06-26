#if MPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(32u);
ForAllChannelsWithAlphaInvokeDivSrcCScale(32u);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(32u);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(32u);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(32u);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(32u);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(32u);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(32u);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
