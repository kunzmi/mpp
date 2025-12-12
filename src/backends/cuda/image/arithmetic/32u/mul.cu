#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(32u);
ForAllChannelsWithAlphaInvokeMulSrcSrcScale(32u);
ForAllChannelsWithAlphaInvokeMulSrcC(32u);
ForAllChannelsWithAlphaInvokeMulSrcCScale(32u);
ForAllChannelsWithAlphaInvokeMulSrcDevC(32u);
ForAllChannelsWithAlphaInvokeMulSrcDevCScale(32u);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScale(32u);
ForAllChannelsWithAlphaInvokeMulInplaceC(32u);
ForAllChannelsWithAlphaInvokeMulInplaceCScale(32u);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(32u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScale(32u);

} // namespace mpp::image::cuda
