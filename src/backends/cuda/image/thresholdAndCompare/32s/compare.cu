#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(32s);
ForAllChannelsWithAlphaInvokeCompareSrcC(32s);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(32s);

ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(32s);
ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(32s);
ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
