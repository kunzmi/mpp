#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcCMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcCScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(16s);
ForAllChannelsWithAlphaInvokeAddSrcDevCScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceCScaleMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(16s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScaleMask(16s);

} // namespace mpp::image::cuda
