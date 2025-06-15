#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
