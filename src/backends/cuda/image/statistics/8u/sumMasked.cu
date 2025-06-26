#if MPP_ENABLE_CUDA_BACKEND

#include "../sumMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8u, 1);
ForAllChannelsWithAlpha(8u, 2);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
