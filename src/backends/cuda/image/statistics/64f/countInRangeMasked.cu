#if OPP_ENABLE_CUDA_BACKEND

#include "../countInRangeMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(64f, 64u, 64u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
