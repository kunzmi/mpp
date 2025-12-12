#include "../sub_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(8s);
ForAllChannelsWithAlphaInvokeSubSrcSrcScale(8s);
ForAllChannelsWithAlphaInvokeSubSrcC(8s);
ForAllChannelsWithAlphaInvokeSubSrcCScale(8s);
ForAllChannelsWithAlphaInvokeSubSrcDevC(8s);
ForAllChannelsWithAlphaInvokeSubSrcDevCScale(8s);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScale(8s);
ForAllChannelsWithAlphaInvokeSubInplaceC(8s);
ForAllChannelsWithAlphaInvokeSubInplaceCScale(8s);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(8s);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScale(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScale(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScale(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(8s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScale(8s);

} // namespace mpp::image::cuda
