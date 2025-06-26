#if MPP_ENABLE_CUDA_BACKEND

#include "../integralSqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32u, 32s, 32s);
ForAllChannelsNoAlpha(32u, 32f, 64f);
ForAllChannelsNoAlpha(32u, 32s, 64s);
ForAllChannelsNoAlpha(32u, 64f, 64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
