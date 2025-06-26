#if MPP_ENABLE_CUDA_BACKEND

#include "../integral_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16bf, 32f);
ForAllChannelsNoAlpha(16bf, 64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
