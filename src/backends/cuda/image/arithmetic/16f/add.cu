#if OPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(16f);
ForAllChannelsWithAlphaInvokeAddSrcC(16f);
ForAllChannelsWithAlphaInvokeAddSrcDevC(16f);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeAddInplaceC(16f);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
