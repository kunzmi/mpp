#if MPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(8s);
ForAllChannelsWithAlphaInvokeAddSrcSrcScale(8s);
ForAllChannelsWithAlphaInvokeAddSrcC(8s);
ForAllChannelsWithAlphaInvokeAddSrcCScale(8s);
ForAllChannelsWithAlphaInvokeAddSrcDevC(8s);
ForAllChannelsWithAlphaInvokeAddSrcDevCScale(8s);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScale(8s);
ForAllChannelsWithAlphaInvokeAddInplaceC(8s);
ForAllChannelsWithAlphaInvokeAddInplaceCScale(8s);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(8s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScale(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
