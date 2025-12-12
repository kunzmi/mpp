#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcCMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcCScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcDevCScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceCScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScaleMask(16u);

} // namespace mpp::image::cuda
