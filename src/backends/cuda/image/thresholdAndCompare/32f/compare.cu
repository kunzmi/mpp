#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(32f);
ForAllChannelsWithAlphaInvokeCompareSrcC(32f);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(32f);
ForAllChannelsWithAlphaInvokeCompareSrc(32f);

ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(32f);
ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(32f);
ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(32f);
ForAllChannelsWithAlphaInvokeCompareSrcAnyChannel(32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
