#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(16u);
ForAllChannelsWithAlphaInvokeCompareSrcC(16u);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(16u);

ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(16u);
ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(16u);
ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(16u);

} // namespace mpp::image::cuda
