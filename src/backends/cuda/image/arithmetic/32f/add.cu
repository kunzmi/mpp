#if OPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(32f);
ForAllChannelsWithAlphaInvokeAddSrcC(32f);
ForAllChannelsWithAlphaInvokeAddSrcDevC(32f);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeAddInplaceC(32f);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
