#if OPP_ENABLE_CUDA_BACKEND

#include "../crossCorrelationCoefficient_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16u, 32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
