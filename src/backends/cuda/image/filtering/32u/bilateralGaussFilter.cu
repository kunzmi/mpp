#if MPP_ENABLE_CUDA_BACKEND

#include "../bilateralGaussFilter_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32u, 32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
