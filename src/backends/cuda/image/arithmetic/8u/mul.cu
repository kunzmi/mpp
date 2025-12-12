#include "../mul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrc(8u);
ForAllChannelsWithAlphaInvokeMulSrcSrcScale(8u);
ForAllChannelsWithAlphaInvokeMulSrcC(8u);
ForAllChannelsWithAlphaInvokeMulSrcCScale(8u);
ForAllChannelsWithAlphaInvokeMulSrcDevC(8u);
ForAllChannelsWithAlphaInvokeMulSrcDevCScale(8u);
ForAllChannelsWithAlphaInvokeMulInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScale(8u);
ForAllChannelsWithAlphaInvokeMulInplaceC(8u);
ForAllChannelsWithAlphaInvokeMulInplaceCScale(8u);
ForAllChannelsWithAlphaInvokeMulInplaceDevC(8u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScale(8u);

} // namespace mpp::image::cuda
