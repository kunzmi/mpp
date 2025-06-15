#if OPP_ENABLE_CUDA_BACKEND

#include "../rectStdDev_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32f, 64f, 64f, 32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
