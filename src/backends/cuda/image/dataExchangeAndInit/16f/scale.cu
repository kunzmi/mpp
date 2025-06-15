#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16f, 8s);
ForAllChannelsWithAlpha(16f, 8u);
ForAllChannelsWithAlpha(16f, 16s);
ForAllChannelsWithAlpha(16f, 16u);
ForAllChannelsWithAlpha(16f, 32u);
ForAllChannelsWithAlpha(16f, 32s);
ForAllChannelsWithAlpha(16f, 32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
