#if MPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeLnSrc(16sc);
ForAllChannelsNoAlphaInvokeLnInplace(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
