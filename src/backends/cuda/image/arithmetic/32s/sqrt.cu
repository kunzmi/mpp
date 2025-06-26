#if MPP_ENABLE_CUDA_BACKEND

#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(32s);
ForAllChannelsWithAlphaInvokeSqrtInplace(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
