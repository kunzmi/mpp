#if MPP_ENABLE_CUDA_BACKEND

#include "../makeComplex_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMakeComplexSrc(32f, 32fc);
ForAllChannelsNoAlphaInvokeMakeComplexSrcSrc(32f, 32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
