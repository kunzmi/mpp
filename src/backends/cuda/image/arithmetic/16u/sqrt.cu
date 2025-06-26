#if MPP_ENABLE_CUDA_BACKEND

#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(16u);
ForAllChannelsWithAlphaInvokeSqrtInplace(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
