#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(8s);
ForAllChannelsWithAlphaInvokeMulSrcSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeMulSrcCMask(8s);
ForAllChannelsWithAlphaInvokeMulSrcCScaleMask(8s);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(8s);
ForAllChannelsWithAlphaInvokeMulSrcDevCScaleMask(8s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(8s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(8s);
ForAllChannelsWithAlphaInvokeMulInplaceCScaleMask(8s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(8s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScaleMask(8s);

} // namespace mpp::image::cuda
