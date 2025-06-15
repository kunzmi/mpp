#if OPP_ENABLE_CUDA_BACKEND

#include "../countInRange_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8u, 64u, 64u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
