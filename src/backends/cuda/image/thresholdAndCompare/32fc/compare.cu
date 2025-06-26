#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeCompareSrcC(32fc);
ForAllChannelsNoAlphaInvokeCompareSrcDevC(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
