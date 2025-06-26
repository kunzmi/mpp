#if MPP_ENABLE_CUDA_BACKEND

#include "../not_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeNotSrc(32s);
ForAllChannelsWithAlphaInvokeNotInplace(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
