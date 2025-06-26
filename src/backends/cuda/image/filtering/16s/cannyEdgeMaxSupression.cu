#if MPP_ENABLE_CUDA_BACKEND

#include "../cannyEdgeMaxSupression_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16s, 8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
