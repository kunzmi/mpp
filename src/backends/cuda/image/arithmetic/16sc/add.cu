#if MPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddSrcSrc(16sc);
ForAllChannelsNoAlphaInvokeAddSrcSrcScale(16sc);
ForAllChannelsNoAlphaInvokeAddSrcC(16sc);
ForAllChannelsNoAlphaInvokeAddSrcCScale(16sc);
ForAllChannelsNoAlphaInvokeAddSrcDevC(16sc);
ForAllChannelsNoAlphaInvokeAddSrcDevCScale(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceSrc(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceSrcScale(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceC(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceCScale(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceDevC(16sc);
ForAllChannelsNoAlphaInvokeAddInplaceDevCScale(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
