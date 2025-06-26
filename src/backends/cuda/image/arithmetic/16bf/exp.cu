#if MPP_ENABLE_CUDA_BACKEND

#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(16bf);
ForAllChannelsWithAlphaInvokeExpInplace(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
