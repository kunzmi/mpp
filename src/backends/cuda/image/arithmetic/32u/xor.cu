#if OPP_ENABLE_CUDA_BACKEND

#include "../xor_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(32u);
ForAllChannelsWithAlphaInvokeXorSrcC(32u);
ForAllChannelsWithAlphaInvokeXorSrcDevC(32u);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeXorInplaceC(32u);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
