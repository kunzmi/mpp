#if MPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(16bf);
ForAllChannelsWithAlphaInvokeLnInplace(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
