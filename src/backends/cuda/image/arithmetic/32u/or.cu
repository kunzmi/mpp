#include "../or_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(32u);
ForAllChannelsWithAlphaInvokeOrSrcC(32u);
ForAllChannelsWithAlphaInvokeOrSrcDevC(32u);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeOrInplaceC(32u);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(32u);

} // namespace mpp::image::cuda
