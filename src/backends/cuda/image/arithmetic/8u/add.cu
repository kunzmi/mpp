#if OPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(8u);
ForAllChannelsWithAlphaInvokeAddSrcSrcScale(8u);
ForAllChannelsWithAlphaInvokeAddSrcC(8u);
ForAllChannelsWithAlphaInvokeAddSrcCScale(8u);
ForAllChannelsWithAlphaInvokeAddSrcDevC(8u);
ForAllChannelsWithAlphaInvokeAddSrcDevCScale(8u);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScale(8u);
ForAllChannelsWithAlphaInvokeAddInplaceC(8u);
ForAllChannelsWithAlphaInvokeAddInplaceCScale(8u);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(8u);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScale(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
