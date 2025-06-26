#if MPP_ENABLE_CUDA_BACKEND

#include "../conj_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjSrc(32sc);
ForAllChannelsNoAlphaInvokeConjInplace(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
