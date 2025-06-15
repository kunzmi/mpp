#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcMask(64f);
ForAllChannelsWithAlphaInvokeDivSrcCMask(64f);
ForAllChannelsWithAlphaInvokeDivSrcDevCMask(64f);
ForAllChannelsWithAlphaInvokeDivInplaceSrcMask(64f);
ForAllChannelsWithAlphaInvokeDivInplaceCMask(64f);
ForAllChannelsWithAlphaInvokeDivInplaceDevCMask(64f);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcMask(64f);
ForAllChannelsWithAlphaInvokeDivInvInplaceCMask(64f);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCMask(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
