#if OPP_ENABLE_CUDA_BACKEND

#include "../columnWindowSum_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
