#if MPP_ENABLE_CUDA_BACKEND

#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(32s);
ForAllChannelsWithAlphaInvokeExpInplace(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
