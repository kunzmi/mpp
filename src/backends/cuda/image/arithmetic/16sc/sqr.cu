#if OPP_ENABLE_CUDA_BACKEND

#include "../sqr_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSqrSrc(16sc);
ForAllChannelsNoAlphaInvokeSqrInplace(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
