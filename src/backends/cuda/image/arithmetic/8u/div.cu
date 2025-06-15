#if OPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(8u);
ForAllChannelsWithAlphaInvokeDivSrcCScale(8u);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(8u);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(8u);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(8u);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(8u);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(8u);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(8u);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
