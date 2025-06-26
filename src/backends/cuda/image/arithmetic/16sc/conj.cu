#if MPP_ENABLE_CUDA_BACKEND

#include "../conj_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjSrc(16sc);
ForAllChannelsNoAlphaInvokeConjInplace(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
