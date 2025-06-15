#if OPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(64f);
ForAllChannelsWithAlphaInvokeLnInplace(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
