#if OPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeLnSrc(16sc);
ForAllChannelsNoAlphaInvokeLnInplace(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
