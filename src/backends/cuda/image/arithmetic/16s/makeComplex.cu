#if MPP_ENABLE_CUDA_BACKEND

#include "../makeComplex_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMakeComplexSrc(16s, 16sc);
ForAllChannelsNoAlphaInvokeMakeComplexSrcSrc(16s, 16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
