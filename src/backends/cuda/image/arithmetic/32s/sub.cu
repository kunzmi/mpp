#include "../sub_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(32s);
ForAllChannelsWithAlphaInvokeSubSrcSrcScale(32s);
ForAllChannelsWithAlphaInvokeSubSrcC(32s);
ForAllChannelsWithAlphaInvokeSubSrcCScale(32s);
ForAllChannelsWithAlphaInvokeSubSrcDevC(32s);
ForAllChannelsWithAlphaInvokeSubSrcDevCScale(32s);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeSubInplaceSrcScale(32s);
ForAllChannelsWithAlphaInvokeSubInplaceC(32s);
ForAllChannelsWithAlphaInvokeSubInplaceCScale(32s);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(32s);
ForAllChannelsWithAlphaInvokeSubInplaceDevCScale(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScale(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceCScale(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(32s);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScale(32s);

} // namespace mpp::image::cuda
