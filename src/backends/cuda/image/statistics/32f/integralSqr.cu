#if OPP_ENABLE_CUDA_BACKEND

#include "../integralSqr_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32f, 32f, 64f);
ForAllChannelsNoAlpha(32f, 64f, 64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
