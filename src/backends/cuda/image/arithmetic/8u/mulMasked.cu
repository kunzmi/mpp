#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(8u);
ForAllChannelsWithAlphaInvokeMulSrcSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeMulSrcCMask(8u);
ForAllChannelsWithAlphaInvokeMulSrcCScaleMask(8u);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(8u);
ForAllChannelsWithAlphaInvokeMulSrcDevCScaleMask(8u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(8u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(8u);
ForAllChannelsWithAlphaInvokeMulInplaceCScaleMask(8u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(8u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScaleMask(8u);

} // namespace mpp::image::cuda
