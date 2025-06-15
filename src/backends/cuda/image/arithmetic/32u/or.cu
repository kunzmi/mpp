#if OPP_ENABLE_CUDA_BACKEND

#include "../or_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(32u);
ForAllChannelsWithAlphaInvokeOrSrcC(32u);
ForAllChannelsWithAlphaInvokeOrSrcDevC(32u);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeOrInplaceC(32u);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
