#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeCompareSrcC(16bf);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeCompareSrc(16bf);

ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(16bf);
ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(16bf);
ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(16bf);
ForAllChannelsWithAlphaInvokeCompareSrcAnyChannel(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
