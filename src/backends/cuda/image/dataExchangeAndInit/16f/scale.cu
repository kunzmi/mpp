#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16f, 8s);
ForAllChannelsWithAlpha(16f, 8u);
ForAllChannelsWithAlpha(16f, 16s);
ForAllChannelsWithAlpha(16f, 16u);
ForAllChannelsWithAlpha(16f, 32u);
ForAllChannelsWithAlpha(16f, 32s);
ForAllChannelsWithAlpha(16f, 32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
