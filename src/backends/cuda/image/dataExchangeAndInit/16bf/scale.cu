#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaIntRound(16bf, 8s);
ForAllChannelsWithAlphaIntRound(16bf, 8u);
ForAllChannelsWithAlphaIntRound(16bf, 16s);
ForAllChannelsWithAlphaIntRound(16bf, 16u);
ForAllChannelsWithAlphaIntRound(16bf, 32u);
ForAllChannelsWithAlphaIntRound(16bf, 32s);
ForAllChannelsWithAlphaFloat(16bf, 16f);
ForAllChannelsWithAlphaFloat(16bf, 32f);
ForAllChannelsWithAlphaFloat(16bf, 64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
