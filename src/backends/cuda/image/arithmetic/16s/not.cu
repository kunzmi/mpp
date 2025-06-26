#if MPP_ENABLE_CUDA_BACKEND

#include "../not_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeNotSrc(16s);
ForAllChannelsWithAlphaInvokeNotInplace(16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
