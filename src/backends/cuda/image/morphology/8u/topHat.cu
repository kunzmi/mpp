#if MPP_ENABLE_CUDA_BACKEND

#include "../topHat_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8u, 8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
