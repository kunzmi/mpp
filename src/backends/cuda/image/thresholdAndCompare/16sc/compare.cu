#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareSrcSrc(16sc);
ForAllChannelsNoAlphaInvokeCompareSrcC(16sc);
ForAllChannelsNoAlphaInvokeCompareSrcDevC(16sc);

ForAllChannelsNoAlphaInvokeCompareSrcSrcAnyChannel(16sc);
ForAllChannelsNoAlphaInvokeCompareSrcCAnyChannel(16sc);
ForAllChannelsNoAlphaInvokeCompareSrcDevCAnyChannel(16sc);

} // namespace mpp::image::cuda
