#if MPP_ENABLE_CUDA_BACKEND

#include "../fixedSizeSeparableFilter_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16f, float);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
