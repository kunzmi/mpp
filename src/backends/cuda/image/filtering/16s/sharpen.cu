#if MPP_ENABLE_CUDA_BACKEND

#include "../sharpen_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16s, 16s);
ForAllChannelsWithAlpha(16s, 32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
