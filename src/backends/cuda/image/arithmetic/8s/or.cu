#if OPP_ENABLE_CUDA_BACKEND

#include "../or_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(8s);
ForAllChannelsWithAlphaInvokeOrSrcC(8s);
ForAllChannelsWithAlphaInvokeOrSrcDevC(8s);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeOrInplaceC(8s);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
