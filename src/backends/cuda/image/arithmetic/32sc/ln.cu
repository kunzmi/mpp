#if OPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeLnSrc(32sc);
ForAllChannelsNoAlphaInvokeLnInplace(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
