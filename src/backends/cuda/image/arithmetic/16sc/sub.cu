#include "../sub_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSubSrcSrc(16sc);
ForAllChannelsNoAlphaInvokeSubSrcSrcScale(16sc);
ForAllChannelsNoAlphaInvokeSubSrcC(16sc);
ForAllChannelsNoAlphaInvokeSubSrcCScale(16sc);
ForAllChannelsNoAlphaInvokeSubSrcDevC(16sc);
ForAllChannelsNoAlphaInvokeSubSrcDevCScale(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceSrc(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceSrcScale(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceC(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceCScale(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceDevC(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceDevCScale(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrc(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrcScale(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceC(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceCScale(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevC(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevCScale(16sc);

} // namespace mpp::image::cuda
