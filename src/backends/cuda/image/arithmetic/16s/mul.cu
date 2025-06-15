#if OPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(16s);
ForAllChannelsWithAlphaInvokeMulSrcSrcScale(16s);
ForAllChannelsWithAlphaInvokeMulSrcC(16s);
ForAllChannelsWithAlphaInvokeMulSrcCScale(16s);
ForAllChannelsWithAlphaInvokeMulSrcDevC(16s);
ForAllChannelsWithAlphaInvokeMulSrcDevCScale(16s);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScale(16s);
ForAllChannelsWithAlphaInvokeMulInplaceC(16s);
ForAllChannelsWithAlphaInvokeMulInplaceCScale(16s);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(16s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScale(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
