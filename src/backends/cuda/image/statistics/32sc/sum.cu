#if OPP_ENABLE_CUDA_BACKEND

#include "../sum_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32sc, 1);
ForAllChannelsNoAlpha(32sc, 2);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
