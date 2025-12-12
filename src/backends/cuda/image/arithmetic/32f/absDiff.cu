#include "../absDiff_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(32f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(32f);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(32f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(32f);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(32f);

} // namespace mpp::image::cuda
