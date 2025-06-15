#if OPP_ENABLE_CUDA_BACKEND

#include "../conjMul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjMulSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeConjMulInplaceSrc(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
