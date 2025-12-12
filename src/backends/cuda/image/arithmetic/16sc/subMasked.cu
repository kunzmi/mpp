#include "../subMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSubSrcSrcMask(16sc);
ForAllChannelsNoAlphaInvokeSubSrcSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeSubSrcCMask(16sc);
ForAllChannelsNoAlphaInvokeSubSrcCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeSubSrcDevCMask(16sc);
ForAllChannelsNoAlphaInvokeSubSrcDevCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceSrcMask(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceCMask(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceDevCMask(16sc);
ForAllChannelsNoAlphaInvokeSubInplaceDevCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrcMask(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceCMask(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevCMask(16sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevCScaleMask(16sc);

} // namespace mpp::image::cuda
