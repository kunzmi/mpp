#if MPP_ENABLE_CUDA_BACKEND

#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaIntRound(32fc, 16sc);
ForAllChannelsNoAlphaIntRound(32fc, 32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
