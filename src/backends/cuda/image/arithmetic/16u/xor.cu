#if OPP_ENABLE_CUDA_BACKEND

#include "../xor_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(16u);
ForAllChannelsWithAlphaInvokeXorSrcC(16u);
ForAllChannelsWithAlphaInvokeXorSrcDevC(16u);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeXorInplaceC(16u);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
