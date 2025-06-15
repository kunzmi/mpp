#if OPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(8s);
ForAllChannelsWithAlphaInvokeMulSrcSrcScale(8s);
ForAllChannelsWithAlphaInvokeMulSrcC(8s);
ForAllChannelsWithAlphaInvokeMulSrcCScale(8s);
ForAllChannelsWithAlphaInvokeMulSrcDevC(8s);
ForAllChannelsWithAlphaInvokeMulSrcDevCScale(8s);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScale(8s);
ForAllChannelsWithAlphaInvokeMulInplaceC(8s);
ForAllChannelsWithAlphaInvokeMulInplaceCScale(8s);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(8s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScale(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
