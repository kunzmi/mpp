#include "../and_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(8s);
ForAllChannelsWithAlphaInvokeAndSrcC(8s);
ForAllChannelsWithAlphaInvokeAndSrcDevC(8s);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeAndInplaceC(8s);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(8s);

} // namespace mpp::image::cuda
