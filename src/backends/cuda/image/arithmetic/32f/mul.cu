#if OPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(32f);
ForAllChannelsWithAlphaInvokeMulSrcC(32f);
ForAllChannelsWithAlphaInvokeMulSrcDevC(32f);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeMulInplaceC(32f);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
