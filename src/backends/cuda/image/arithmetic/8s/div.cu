#if OPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(8s);
ForAllChannelsWithAlphaInvokeDivSrcCScale(8s);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(8s);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(8s);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(8s);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(8s);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(8s);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(8s);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
