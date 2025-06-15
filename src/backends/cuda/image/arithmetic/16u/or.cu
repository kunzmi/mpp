#if OPP_ENABLE_CUDA_BACKEND

#include "../or_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(16u);
ForAllChannelsWithAlphaInvokeOrSrcC(16u);
ForAllChannelsWithAlphaInvokeOrSrcDevC(16u);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeOrInplaceC(16u);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
