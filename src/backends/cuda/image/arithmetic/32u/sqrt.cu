#if OPP_ENABLE_CUDA_BACKEND

#include "../sqrt_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(32u);
ForAllChannelsWithAlphaInvokeSqrtInplace(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
