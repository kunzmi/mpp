#if OPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(16s);
ForAllChannelsWithAlphaInvokeDivSrcCScale(16s);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(16s);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(16s);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(16s);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(16s);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(16s);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(16s);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
