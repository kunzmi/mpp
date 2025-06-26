#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32u, 8s);
ForAllChannelsWithAlpha(32u, 8u);
ForAllChannelsWithAlpha(32u, 16s);
ForAllChannelsWithAlpha(32u, 16u);
ForAllChannelsWithAlpha(32u, 32s);
ForAllChannelsWithAlpha(32u, 32f);
ForAllChannelsWithAlpha(32u, 64f);
ForAllChannelsWithAlpha(32u, 16f);
ForAllChannelsWithAlpha(32u, 16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
