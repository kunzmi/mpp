#if MPP_ENABLE_CUDA_BACKEND

#include "../normInf_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
