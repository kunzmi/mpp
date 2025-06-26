#if MPP_ENABLE_CUDA_BACKEND

#include "../abs_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsSrc(64f);
ForAllChannelsWithAlphaInvokeAbsInplace(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
