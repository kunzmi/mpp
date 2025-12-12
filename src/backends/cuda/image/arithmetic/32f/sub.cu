#include "../sub_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrc(32f);
ForAllChannelsWithAlphaInvokeSubSrcC(32f);
ForAllChannelsWithAlphaInvokeSubSrcDevC(32f);
ForAllChannelsWithAlphaInvokeSubInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeSubInplaceC(32f);
ForAllChannelsWithAlphaInvokeSubInplaceDevC(32f);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeSubInvInplaceC(32f);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(32f);

} // namespace mpp::image::cuda
