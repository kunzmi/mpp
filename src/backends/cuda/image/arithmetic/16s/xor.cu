#include "../xor_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeXorSrcSrc(16s);
ForAllChannelsWithAlphaInvokeXorSrcC(16s);
ForAllChannelsWithAlphaInvokeXorSrcDevC(16s);
ForAllChannelsWithAlphaInvokeXorInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeXorInplaceC(16s);
ForAllChannelsWithAlphaInvokeXorInplaceDevC(16s);

} // namespace mpp::image::cuda
