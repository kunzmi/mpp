#include "../xor_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(32s);
ForAllChannelsWithAlphaInvokeXorSrcC(32s);
ForAllChannelsWithAlphaInvokeXorSrcDevC(32s);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeXorInplaceC(32s);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(32s);

} // namespace mpp::image::cuda
