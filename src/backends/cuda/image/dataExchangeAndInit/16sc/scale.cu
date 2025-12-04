#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaIntDiv(16sc, 32sc);
ForAllChannelsNoAlphaFloat(16sc, 32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
