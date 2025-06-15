#if OPP_ENABLE_CUDA_BACKEND

#include "../sum_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32s, 1);
ForAllChannelsWithAlpha(32s, 2);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
