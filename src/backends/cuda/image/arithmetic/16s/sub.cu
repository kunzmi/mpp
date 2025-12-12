#include "../sub_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(16s);
ForAllChannelsWithAlphaInvokeSubSrcSrcScale(16s);
ForAllChannelsWithAlphaInvokeSubSrcC(16s);
ForAllChannelsWithAlphaInvokeSubSrcCScale(16s);
ForAllChannelsWithAlphaInvokeSubSrcDevC(16s);
ForAllChannelsWithAlphaInvokeSubSrcDevCScale(16s);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScale(16s);
ForAllChannelsWithAlphaInvokeSubInplaceC(16s);
ForAllChannelsWithAlphaInvokeSubInplaceCScale(16s);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(16s);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScale(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScale(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScale(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(16s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScale(16s);

} // namespace mpp::image::cuda
