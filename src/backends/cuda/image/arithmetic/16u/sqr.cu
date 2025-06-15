#if OPP_ENABLE_CUDA_BACKEND

#include "../sqr_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(16u);
ForAllChannelsWithAlphaInvokeSqrInplace(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
