#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16u, 8s);
ForAllChannelsWithAlpha(16u, 8u);
ForAllChannelsWithAlpha(16u, 16s);
ForAllChannelsWithAlpha(16u, 32u);
ForAllChannelsWithAlpha(16u, 32s);
ForAllChannelsWithAlpha(16u, 32f);
ForAllChannelsWithAlpha(16u, 64f);
ForAllChannelsWithAlpha(16u, 16f);
ForAllChannelsWithAlpha(16u, 16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
