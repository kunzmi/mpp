#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(16f);
ForAllChannelsWithAlphaInvokeCompareSrcC(16f);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(16f);
ForAllChannelsWithAlphaInvokeCompareSrc(16f);

ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(16f);
ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(16f);
ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(16f);
ForAllChannelsWithAlphaInvokeCompareSrcAnyChannel(16f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
