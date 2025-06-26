#if MPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeCompareSrcC(32sc);
ForAllChannelsNoAlphaInvokeCompareSrcDevC(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
