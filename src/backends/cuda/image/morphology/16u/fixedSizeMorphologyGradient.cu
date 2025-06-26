#if MPP_ENABLE_CUDA_BACKEND

#include "../fixedSizeMorphologyGradient_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16u, 16u);
ForAllChannelsWithAlpha(16u, 32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
