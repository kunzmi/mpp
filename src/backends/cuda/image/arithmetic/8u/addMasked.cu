#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(8u);
ForAllChannelsWithAlphaInvokeAddSrcSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeAddSrcCMask(8u);
ForAllChannelsWithAlphaInvokeAddSrcCScaleMask(8u);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(8u);
ForAllChannelsWithAlphaInvokeAddSrcDevCScaleMask(8u);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(8u);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(8u);
ForAllChannelsWithAlphaInvokeAddInplaceCScaleMask(8u);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(8u);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScaleMask(8u);

} // namespace mpp::image::cuda
