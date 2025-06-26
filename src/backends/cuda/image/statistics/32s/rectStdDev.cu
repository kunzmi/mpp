#if MPP_ENABLE_CUDA_BACKEND

#include "../rectStdDev_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32s, 32s, 64f, 32f);
ForAllChannelsNoAlpha(32s, 64s, 64f, 32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
