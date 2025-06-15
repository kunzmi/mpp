#if OPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(16f);
ForAllChannelsWithAlphaInvokeMulSrcC(16f);
ForAllChannelsWithAlphaInvokeMulSrcDevC(16f);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeMulInplaceC(16f);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
