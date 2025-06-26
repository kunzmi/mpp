#if MPP_ENABLE_CUDA_BACKEND

#include "../separableFilter_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16bf, float);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
