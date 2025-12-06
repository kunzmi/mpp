#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(32u);
ForAllChannelsWithAlphaInvokeCompareSrcC(32u);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(32u);

ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(32u);
ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(32u);
ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
