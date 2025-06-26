#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8s, 8u);
ForAllChannelsWithAlpha(8s, 16u);
ForAllChannelsWithAlpha(8s, 16s);
ForAllChannelsWithAlpha(8s, 32u);
ForAllChannelsWithAlpha(8s, 32s);
ForAllChannelsWithAlpha(8s, 32f);
ForAllChannelsWithAlpha(8s, 64f);
ForAllChannelsWithAlpha(8s, 16f);
ForAllChannelsWithAlpha(8s, 16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
