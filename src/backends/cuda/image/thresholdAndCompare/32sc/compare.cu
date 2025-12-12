#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeCompareSrcC(32sc);
ForAllChannelsNoAlphaInvokeCompareSrcDevC(32sc);

ForAllChannelsNoAlphaInvokeCompareSrcSrcAnyChannel(32sc);
ForAllChannelsNoAlphaInvokeCompareSrcCAnyChannel(32sc);
ForAllChannelsNoAlphaInvokeCompareSrcDevCAnyChannel(32sc);

} // namespace mpp::image::cuda
