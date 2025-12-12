#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(32u);
ForAllChannelsWithAlphaInvokeAddSrcSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeAddSrcCMask(32u);
ForAllChannelsWithAlphaInvokeAddSrcCScaleMask(32u);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(32u);
ForAllChannelsWithAlphaInvokeAddSrcDevCScaleMask(32u);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(32u);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(32u);
ForAllChannelsWithAlphaInvokeAddInplaceCScaleMask(32u);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(32u);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScaleMask(32u);

} // namespace mpp::image::cuda
