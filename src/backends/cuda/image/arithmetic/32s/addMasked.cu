#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcCMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcCScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(32s);
ForAllChannelsWithAlphaInvokeAddSrcDevCScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceCScaleMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(32s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScaleMask(32s);

} // namespace mpp::image::cuda
