#include "../or_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeOrSrcSrc(32s);
ForAllChannelsWithAlphaInvokeOrSrcC(32s);
ForAllChannelsWithAlphaInvokeOrSrcDevC(32s);
ForAllChannelsWithAlphaInvokeOrInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeOrInplaceC(32s);
ForAllChannelsWithAlphaInvokeOrInplaceDevC(32s);

} // namespace mpp::image::cuda
