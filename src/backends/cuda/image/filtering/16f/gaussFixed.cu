#if MPP_ENABLE_CUDA_BACKEND

#include "../gaussFixed_impl.h"

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
