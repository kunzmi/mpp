#if OPP_ENABLE_CUDA_BACKEND

#include "../xor_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(8s);
ForAllChannelsWithAlphaInvokeXorSrcC(8s);
ForAllChannelsWithAlphaInvokeXorSrcDevC(8s);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeXorInplaceC(8s);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
