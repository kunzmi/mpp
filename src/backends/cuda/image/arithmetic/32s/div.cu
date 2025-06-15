#if OPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(32s);
ForAllChannelsWithAlphaInvokeDivSrcCScale(32s);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(32s);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(32s);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(32s);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(32s);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(32s);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(32s);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
