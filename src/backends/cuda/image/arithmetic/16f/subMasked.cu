#include "../subMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(16f);
ForAllChannelsWithAlphaInvokeSubSrcCMask(16f);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(16f);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(16f);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(16f);

} // namespace mpp::image::cuda
