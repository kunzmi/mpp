#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
