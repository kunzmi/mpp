#if MPP_ENABLE_CUDA_BACKEND

#include "../imag_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16sc, 16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
