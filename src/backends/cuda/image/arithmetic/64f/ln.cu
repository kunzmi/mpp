#if MPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(64f);
ForAllChannelsWithAlphaInvokeLnInplace(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
