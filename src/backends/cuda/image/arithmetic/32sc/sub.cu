#if OPP_ENABLE_CUDA_BACKEND

#include "../sub_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSubSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeSubSrcSrcScale(32sc);
ForAllChannelsNoAlphaInvokeSubSrcC(32sc);
ForAllChannelsNoAlphaInvokeSubSrcCScale(32sc);
ForAllChannelsNoAlphaInvokeSubSrcDevC(32sc);
ForAllChannelsNoAlphaInvokeSubSrcDevCScale(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceSrc(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceSrcScale(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceC(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceCScale(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceDevC(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceDevCScale(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrc(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrcScale(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceC(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceCScale(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevC(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevCScale(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
