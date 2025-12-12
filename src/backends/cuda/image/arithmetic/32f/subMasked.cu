#include "../subMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSubSrcSrcMask(32f);
ForAllChannelsWithAlphaInvokeSubSrcCMask(32f);
ForAllChannelsWithAlphaInvokeSubSrcDevCMask(32f);
ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(32f);
ForAllChannelsWithAlphaInvokeSubInplaceCMask(32f);
ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(32f);
ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(32f);
ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(32f);
ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(32f);

} // namespace mpp::image::cuda
