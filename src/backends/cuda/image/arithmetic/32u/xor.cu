#include "../xor_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(32u);
ForAllChannelsWithAlphaInvokeXorSrcC(32u);
ForAllChannelsWithAlphaInvokeXorSrcDevC(32u);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeXorInplaceC(32u);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(32u);

} // namespace mpp::image::cuda
