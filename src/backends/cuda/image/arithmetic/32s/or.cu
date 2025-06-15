#if OPP_ENABLE_CUDA_BACKEND

#include "../or_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(32s);
ForAllChannelsWithAlphaInvokeOrSrcC(32s);
ForAllChannelsWithAlphaInvokeOrSrcDevC(32s);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeOrInplaceC(32s);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
