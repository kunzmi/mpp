#if MPP_ENABLE_CUDA_BACKEND

#include "../countInRangeMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32f, 64u, 64u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
