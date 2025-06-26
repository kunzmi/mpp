#if MPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeLnSrc(32sc);
ForAllChannelsNoAlphaInvokeLnInplace(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
