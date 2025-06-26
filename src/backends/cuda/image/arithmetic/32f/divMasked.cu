#if MPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcMask(32f);
ForAllChannelsWithAlphaInvokeDivSrcCMask(32f);
ForAllChannelsWithAlphaInvokeDivSrcDevCMask(32f);
ForAllChannelsWithAlphaInvokeDivInplaceSrcMask(32f);
ForAllChannelsWithAlphaInvokeDivInplaceCMask(32f);
ForAllChannelsWithAlphaInvokeDivInplaceDevCMask(32f);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcMask(32f);
ForAllChannelsWithAlphaInvokeDivInvInplaceCMask(32f);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCMask(32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
