#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
