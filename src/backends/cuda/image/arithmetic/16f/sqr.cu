#if OPP_ENABLE_CUDA_BACKEND

#include "../sqr_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(16f);
ForAllChannelsWithAlphaInvokeSqrInplace(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
