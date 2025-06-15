#if OPP_ENABLE_CUDA_BACKEND

#include "../xor_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(8u);
ForAllChannelsWithAlphaInvokeXorSrcC(8u);
ForAllChannelsWithAlphaInvokeXorSrcDevC(8u);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeXorInplaceC(8u);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
