#if MPP_ENABLE_CUDA_BACKEND

#include "../fixedSizeBoxFilter_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8s, 8s);
ForAllChannelsWithAlpha(8s, 32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
