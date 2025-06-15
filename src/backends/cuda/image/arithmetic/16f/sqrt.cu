#if OPP_ENABLE_CUDA_BACKEND

#include "../sqrt_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(16f);
ForAllChannelsWithAlphaInvokeSqrtInplace(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
