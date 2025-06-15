#if OPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(64f);
ForAllChannelsWithAlphaInvokeMulSrcC(64f);
ForAllChannelsWithAlphaInvokeMulSrcDevC(64f);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeMulInplaceC(64f);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
