#if MPP_ENABLE_CUDA_BACKEND

#include "../crossCorrelationNormalized_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32s, 32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
