#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
