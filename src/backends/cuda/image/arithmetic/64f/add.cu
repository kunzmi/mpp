#if OPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(64f);
ForAllChannelsWithAlphaInvokeAddSrcC(64f);
ForAllChannelsWithAlphaInvokeAddSrcDevC(64f);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeAddInplaceC(64f);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
