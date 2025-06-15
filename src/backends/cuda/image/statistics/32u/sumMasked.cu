#if OPP_ENABLE_CUDA_BACKEND

#include "../sumMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32u, 1);
ForAllChannelsWithAlpha(32u, 2);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
