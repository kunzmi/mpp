#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareSrcSrc(16sc);
ForAllChannelsNoAlphaInvokeCompareSrcC(16sc);
ForAllChannelsNoAlphaInvokeCompareSrcDevC(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
