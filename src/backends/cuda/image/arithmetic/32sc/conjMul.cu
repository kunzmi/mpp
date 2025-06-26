#if MPP_ENABLE_CUDA_BACKEND

#include "../conjMul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjMulSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeConjMulInplaceSrc(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
