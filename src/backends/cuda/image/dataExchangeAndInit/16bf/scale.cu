#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16bf, 8s);
ForAllChannelsWithAlpha(16bf, 8u);
ForAllChannelsWithAlpha(16bf, 16s);
ForAllChannelsWithAlpha(16bf, 16u);
ForAllChannelsWithAlpha(16bf, 32u);
ForAllChannelsWithAlpha(16bf, 32s);
ForAllChannelsWithAlpha(16bf, 32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
