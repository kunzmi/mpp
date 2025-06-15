#if OPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
