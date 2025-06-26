#if MPP_ENABLE_CUDA_BACKEND

#include "../minMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
