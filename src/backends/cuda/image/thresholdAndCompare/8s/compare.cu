#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(8s);
ForAllChannelsWithAlphaInvokeCompareSrcC(8s);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(8s);

ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(8s);
ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(8s);
ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(8s);

} // namespace mpp::image::cuda
