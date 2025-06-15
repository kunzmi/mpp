#if OPP_ENABLE_CUDA_BACKEND

#include "../or_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(16s);
ForAllChannelsWithAlphaInvokeOrSrcC(16s);
ForAllChannelsWithAlphaInvokeOrSrcDevC(16s);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeOrInplaceC(16s);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
