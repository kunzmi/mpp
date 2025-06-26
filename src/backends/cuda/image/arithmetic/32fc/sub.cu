#if MPP_ENABLE_CUDA_BACKEND

#include "../sub_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSubSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeSubSrcC(32fc);
ForAllChannelsNoAlphaInvokeSubSrcDevC(32fc);
ForAllChannelsNoAlphaInvokeSubInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeSubInplaceC(32fc);
ForAllChannelsNoAlphaInvokeSubInplaceDevC(32fc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeSubInvInplaceC(32fc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevC(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
