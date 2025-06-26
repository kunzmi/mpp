#if MPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcMask(16bf);
ForAllChannelsWithAlphaInvokeDivSrcCMask(16bf);
ForAllChannelsWithAlphaInvokeDivSrcDevCMask(16bf);
ForAllChannelsWithAlphaInvokeDivInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeDivInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeDivInplaceDevCMask(16bf) ForAllChannelsWithAlphaInvokeDivInvInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeDivInvInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCMask(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
