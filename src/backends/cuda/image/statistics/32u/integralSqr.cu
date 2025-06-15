#if OPP_ENABLE_CUDA_BACKEND

#include "../integralSqr_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32u, 32s, 32s);
ForAllChannelsNoAlpha(32u, 32f, 64f);
ForAllChannelsNoAlpha(32u, 32s, 64s);
ForAllChannelsNoAlpha(32u, 64f, 64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
