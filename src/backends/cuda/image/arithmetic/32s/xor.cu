#if OPP_ENABLE_CUDA_BACKEND

#include "../xor_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(32s);
ForAllChannelsWithAlphaInvokeXorSrcC(32s);
ForAllChannelsWithAlphaInvokeXorSrcDevC(32s);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeXorInplaceC(32s);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
