#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32s, 8s);
ForAllChannelsWithAlpha(32s, 8u);
ForAllChannelsWithAlpha(32s, 16s);
ForAllChannelsWithAlpha(32s, 16u);
ForAllChannelsWithAlpha(32s, 32u);
ForAllChannelsWithAlpha(32s, 32f);
ForAllChannelsWithAlpha(32s, 64f);
ForAllChannelsWithAlpha(32s, 16f);
ForAllChannelsWithAlpha(32s, 16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
