#if OPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(16s);
ForAllChannelsWithAlphaInvokeAddSrcSrcScale(16s);
ForAllChannelsWithAlphaInvokeAddSrcC(16s);
ForAllChannelsWithAlphaInvokeAddSrcCScale(16s);
ForAllChannelsWithAlphaInvokeAddSrcDevC(16s);
ForAllChannelsWithAlphaInvokeAddSrcDevCScale(16s);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScale(16s);
ForAllChannelsWithAlphaInvokeAddInplaceC(16s);
ForAllChannelsWithAlphaInvokeAddInplaceCScale(16s);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(16s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScale(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
