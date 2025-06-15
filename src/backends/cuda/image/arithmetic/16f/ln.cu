#if OPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(16f);
ForAllChannelsWithAlphaInvokeLnInplace(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
