#if OPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeMulSrcC(16bf);
ForAllChannelsWithAlphaInvokeMulSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceC(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
