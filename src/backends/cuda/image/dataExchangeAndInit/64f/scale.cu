#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(64f, 8s);
ForAllChannelsWithAlpha(64f, 8u);
ForAllChannelsWithAlpha(64f, 16s);
ForAllChannelsWithAlpha(64f, 16u);
ForAllChannelsWithAlpha(64f, 32u);
ForAllChannelsWithAlpha(64f, 32s);
ForAllChannelsWithAlpha(64f, 32f);
ForAllChannelsWithAlpha(64f, 16f);
ForAllChannelsWithAlpha(64f, 16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
