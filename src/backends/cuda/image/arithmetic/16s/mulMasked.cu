#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcCMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcCScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcDevCScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceCScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScaleMask(16s);

} // namespace mpp::image::cuda
