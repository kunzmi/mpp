#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(16f);
ForAllChannelsWithAlphaInvokeMulSrcCMask(16f);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(16f);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(16f);

} // namespace mpp::image::cuda
