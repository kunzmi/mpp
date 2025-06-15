#if OPP_ENABLE_CUDA_BACKEND

#include "../or_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(8u);
ForAllChannelsWithAlphaInvokeOrSrcC(8u);
ForAllChannelsWithAlphaInvokeOrSrcDevC(8u);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeOrInplaceC(8u);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
