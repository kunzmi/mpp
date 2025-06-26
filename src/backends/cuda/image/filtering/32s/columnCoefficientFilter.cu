#if MPP_ENABLE_CUDA_BACKEND

#include "../columnCoefficientFilter_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32s, float);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
