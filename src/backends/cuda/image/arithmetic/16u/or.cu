#include "../or_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(16u);
ForAllChannelsWithAlphaInvokeOrSrcC(16u);
ForAllChannelsWithAlphaInvokeOrSrcDevC(16u);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeOrInplaceC(16u);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(16u);

} // namespace mpp::image::cuda
