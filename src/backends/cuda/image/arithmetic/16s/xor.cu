#if OPP_ENABLE_CUDA_BACKEND

#include "../xor_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(16s);
ForAllChannelsWithAlphaInvokeXorSrcC(16s);
ForAllChannelsWithAlphaInvokeXorSrcDevC(16s);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeXorInplaceC(16s);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
