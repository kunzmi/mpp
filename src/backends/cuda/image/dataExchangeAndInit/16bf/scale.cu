#if OPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16bf, 8s);
ForAllChannelsWithAlpha(16bf, 8u);
ForAllChannelsWithAlpha(16bf, 16s);
ForAllChannelsWithAlpha(16bf, 16u);
ForAllChannelsWithAlpha(16bf, 32u);
ForAllChannelsWithAlpha(16bf, 32s);
ForAllChannelsWithAlpha(16bf, 32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
