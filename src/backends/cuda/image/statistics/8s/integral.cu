#if MPP_ENABLE_CUDA_BACKEND

#include "../integral_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(8s, 32s);
ForAllChannelsNoAlpha(8s, 32f);
ForAllChannelsNoAlpha(8s, 64s);
ForAllChannelsNoAlpha(8s, 64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
