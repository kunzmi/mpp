#if OPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(32u);
ForAllChannelsWithAlphaInvokeLnInplace(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
