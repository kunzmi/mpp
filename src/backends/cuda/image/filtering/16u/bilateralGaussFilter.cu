#if OPP_ENABLE_CUDA_BACKEND

#include "../bilateralGaussFilter_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16u, 16u);
ForAllChannelsWithAlpha(16u, 32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
