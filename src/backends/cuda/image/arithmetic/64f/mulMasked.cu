#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(64f);
ForAllChannelsWithAlphaInvokeMulSrcCMask(64f);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(64f);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(64f);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(64f);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(64f);

} // namespace mpp::image::cuda
