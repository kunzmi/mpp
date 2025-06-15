#if OPP_ENABLE_CUDA_BACKEND

#include "../conjMul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjMulSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeConjMulInplaceSrc(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
