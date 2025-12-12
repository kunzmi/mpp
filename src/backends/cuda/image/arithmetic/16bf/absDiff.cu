#include "../absDiff_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsDiffSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffSrcC(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceC(16bf);
ForAllChannelsWithAlphaInvokeAbsDiffInplaceDevC(16bf);

} // namespace mpp::image::cuda
