#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16s, 8s);
ForAllChannelsWithAlpha(16s, 8u);
ForAllChannelsWithAlpha(16s, 16u);
ForAllChannelsWithAlpha(16s, 32u);
ForAllChannelsWithAlpha(16s, 32s);
ForAllChannelsWithAlpha(16s, 32f);
ForAllChannelsWithAlpha(16s, 64f);
ForAllChannelsWithAlpha(16s, 16f);
ForAllChannelsWithAlpha(16s, 16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
