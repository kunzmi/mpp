#if OPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(16u);
ForAllChannelsWithAlphaInvokeMulSrcSrcScale(16u);
ForAllChannelsWithAlphaInvokeMulSrcC(16u);
ForAllChannelsWithAlphaInvokeMulSrcCScale(16u);
ForAllChannelsWithAlphaInvokeMulSrcDevC(16u);
ForAllChannelsWithAlphaInvokeMulSrcDevCScale(16u);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScale(16u);
ForAllChannelsWithAlphaInvokeMulInplaceC(16u);
ForAllChannelsWithAlphaInvokeMulInplaceCScale(16u);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(16u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScale(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
