#if MPP_ENABLE_CUDA_BACKEND

#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(16s);
ForAllChannelsWithAlphaInvokeLnInplace(16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
