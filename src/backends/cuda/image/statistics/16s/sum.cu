#if MPP_ENABLE_CUDA_BACKEND

#include "../sum_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16s, 1);
ForAllChannelsWithAlpha(16s, 2);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
