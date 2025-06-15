#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
