#if OPP_ENABLE_CUDA_BACKEND

#include "../sqrt_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSqrtSrc(32fc);
ForAllChannelsNoAlphaInvokeSqrtInplace(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
