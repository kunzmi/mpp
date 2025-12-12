#include "../and_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(16u);
ForAllChannelsWithAlphaInvokeAndSrcC(16u);
ForAllChannelsWithAlphaInvokeAndSrcDevC(16u);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeAndInplaceC(16u);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(16u);

} // namespace mpp::image::cuda
