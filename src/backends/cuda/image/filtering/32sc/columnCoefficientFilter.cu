#if MPP_ENABLE_CUDA_BACKEND

#include "../columnCoefficientFilter_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32sc, float);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
