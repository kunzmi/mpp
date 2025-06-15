#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8s, 8u);
ForAllChannelsWithAlpha(8s, 16u);
ForAllChannelsWithAlpha(8s, 16s);
ForAllChannelsWithAlpha(8s, 32u);
ForAllChannelsWithAlpha(8s, 32s);
ForAllChannelsWithAlpha(8s, 32f);
ForAllChannelsWithAlpha(8s, 64f);
ForAllChannelsWithAlpha(8s, 16f);
ForAllChannelsWithAlpha(8s, 16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
