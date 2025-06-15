#if OPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(32u);
ForAllChannelsWithAlphaInvokeAddSrcSrcScale(32u);
ForAllChannelsWithAlphaInvokeAddSrcC(32u);
ForAllChannelsWithAlphaInvokeAddSrcCScale(32u);
ForAllChannelsWithAlphaInvokeAddSrcDevC(32u);
ForAllChannelsWithAlphaInvokeAddSrcDevCScale(32u);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScale(32u);
ForAllChannelsWithAlphaInvokeAddInplaceC(32u);
ForAllChannelsWithAlphaInvokeAddInplaceCScale(32u);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(32u);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScale(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
