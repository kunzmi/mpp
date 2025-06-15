#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcMask(16bf);
ForAllChannelsWithAlphaInvokeDivSrcCMask(16bf);
ForAllChannelsWithAlphaInvokeDivSrcDevCMask(16bf);
ForAllChannelsWithAlphaInvokeDivInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeDivInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeDivInplaceDevCMask(16bf) ForAllChannelsWithAlphaInvokeDivInvInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeDivInvInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCMask(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
