#if OPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(32s);
ForAllChannelsWithAlphaInvokeAddSrcSrcScale(32s);
ForAllChannelsWithAlphaInvokeAddSrcC(32s);
ForAllChannelsWithAlphaInvokeAddSrcCScale(32s);
ForAllChannelsWithAlphaInvokeAddSrcDevC(32s);
ForAllChannelsWithAlphaInvokeAddSrcDevCScale(32s);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScale(32s);
ForAllChannelsWithAlphaInvokeAddInplaceC(32s);
ForAllChannelsWithAlphaInvokeAddInplaceCScale(32s);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(32s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScale(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
