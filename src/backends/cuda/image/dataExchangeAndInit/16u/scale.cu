#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16u, 8s);
ForAllChannelsWithAlpha(16u, 8u);
ForAllChannelsWithAlpha(16u, 16s);
ForAllChannelsWithAlpha(16u, 32u);
ForAllChannelsWithAlpha(16u, 32s);
ForAllChannelsWithAlpha(16u, 32f);
ForAllChannelsWithAlpha(16u, 64f);
ForAllChannelsWithAlpha(16u, 16f);
ForAllChannelsWithAlpha(16u, 16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
