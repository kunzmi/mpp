#if MPP_ENABLE_CUDA_BACKEND

#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(16f);
ForAllChannelsWithAlphaInvokeSqrInplace(16f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
