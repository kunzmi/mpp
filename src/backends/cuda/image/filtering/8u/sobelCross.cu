#if MPP_ENABLE_CUDA_BACKEND

#include "../sobelCross_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8u, 8u);
ForAllChannelsWithAlpha(8u, 16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
