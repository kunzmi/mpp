#include "../and_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(8u);
ForAllChannelsWithAlphaInvokeAndSrcC(8u);
ForAllChannelsWithAlphaInvokeAndSrcDevC(8u);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeAndInplaceC(8u);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(8u);

} // namespace mpp::image::cuda
