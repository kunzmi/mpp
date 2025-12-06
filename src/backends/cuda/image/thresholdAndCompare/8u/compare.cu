#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(8u);
ForAllChannelsWithAlphaInvokeCompareSrcC(8u);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(8u);

ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(8u);
ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(8u);
ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
