#include "../subMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSubSrcSrcMask(32sc);
ForAllChannelsNoAlphaInvokeSubSrcSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeSubSrcCMask(32sc);
ForAllChannelsNoAlphaInvokeSubSrcCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeSubSrcDevCMask(32sc);
ForAllChannelsNoAlphaInvokeSubSrcDevCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceSrcMask(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceCMask(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceDevCMask(32sc);
ForAllChannelsNoAlphaInvokeSubInplaceDevCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrcMask(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceCMask(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevCMask(32sc);
ForAllChannelsNoAlphaInvokeSubInvInplaceDevCScaleMask(32sc);

} // namespace mpp::image::cuda
