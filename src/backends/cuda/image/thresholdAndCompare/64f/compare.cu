#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(64f);
ForAllChannelsWithAlphaInvokeCompareSrcC(64f);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
