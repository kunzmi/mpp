#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8u, 8s);
ForAllChannelsWithAlpha(8u, 16u);
ForAllChannelsWithAlpha(8u, 16s);
ForAllChannelsWithAlpha(8u, 32u);
ForAllChannelsWithAlpha(8u, 32s);
ForAllChannelsWithAlpha(8u, 32f);
ForAllChannelsWithAlpha(8u, 64f);
ForAllChannelsWithAlpha(8u, 16f);
ForAllChannelsWithAlpha(8u, 16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
