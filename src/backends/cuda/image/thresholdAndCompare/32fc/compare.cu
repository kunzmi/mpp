#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeCompareSrcC(32fc);
ForAllChannelsNoAlphaInvokeCompareSrcDevC(32fc);
ForAllChannelsNoAlphaInvokeCompareSrc(32fc);

ForAllChannelsNoAlphaInvokeCompareSrcSrcAnyChannel(32fc);
ForAllChannelsNoAlphaInvokeCompareSrcCAnyChannel(32fc);
ForAllChannelsNoAlphaInvokeCompareSrcDevCAnyChannel(32fc);
ForAllChannelsNoAlphaInvokeCompareSrcAnyChannel(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
