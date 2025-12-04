#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntDiv(32s, 8s);
ForAllChannelsWithAlphaIntDiv(32s, 8u);
ForAllChannelsWithAlphaIntDiv(32s, 16s);
ForAllChannelsWithAlphaIntDiv(32s, 16u);
ForAllChannelsWithAlphaIntDiv(32s, 32u);
ForAllChannelsWithAlphaFloat(32s, 32f);
ForAllChannelsWithAlphaFloat(32s, 64f);
ForAllChannelsWithAlphaFloat(32s, 16f);
ForAllChannelsWithAlphaFloat(32s, 16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
