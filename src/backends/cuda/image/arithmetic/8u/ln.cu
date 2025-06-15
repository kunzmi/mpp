#if OPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(8u);
ForAllChannelsWithAlphaInvokeLnInplace(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
