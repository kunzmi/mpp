#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcMask(16f);
ForAllChannelsWithAlphaInvokeDivSrcCMask(16f);
ForAllChannelsWithAlphaInvokeDivSrcDevCMask(16f);
ForAllChannelsWithAlphaInvokeDivInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeDivInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeDivInplaceDevCMask(16f);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeDivInvInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCMask(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
