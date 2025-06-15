#if OPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeAddSrcC(16bf);
ForAllChannelsWithAlphaInvokeAddSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceC(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
