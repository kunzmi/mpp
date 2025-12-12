#include "../subMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSubSrcSrcMask(32fc);
ForAllChannelsNoAlphaInvokeSubSrcCMask(32fc);
ForAllChannelsNoAlphaInvokeSubSrcDevCMask(32fc);
ForAllChannelsNoAlphaInvokeSubInplaceSrcMask(32fc);
ForAllChannelsNoAlphaInvokeSubInplaceCMask(32fc);
ForAllChannelsNoAlphaInvokeSubInplaceDevCMask(32fc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrcMask(32fc);
ForAllChannelsNoAlphaInvokeSubInvInplaceCMask(32fc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevCMask(32fc);

} // namespace mpp::image::cuda
