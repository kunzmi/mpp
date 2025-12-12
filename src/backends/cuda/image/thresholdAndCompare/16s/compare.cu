#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(16s);
ForAllChannelsWithAlphaInvokeCompareSrcC(16s);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(16s);

ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(16s);
ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(16s);
ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(16s);

} // namespace mpp::image::cuda
