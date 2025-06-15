#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32f, 8s);
ForAllChannelsWithAlpha(32f, 8u);
ForAllChannelsWithAlpha(32f, 16s);
ForAllChannelsWithAlpha(32f, 16u);
ForAllChannelsWithAlpha(32f, 32u);
ForAllChannelsWithAlpha(32f, 32s);
ForAllChannelsWithAlpha(32f, 64f);
ForAllChannelsWithAlpha(32f, 16f);
ForAllChannelsWithAlpha(32f, 16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
