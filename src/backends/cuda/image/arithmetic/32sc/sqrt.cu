#if MPP_ENABLE_CUDA_BACKEND

#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSqrtSrc(32sc);
ForAllChannelsNoAlphaInvokeSqrtInplace(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
